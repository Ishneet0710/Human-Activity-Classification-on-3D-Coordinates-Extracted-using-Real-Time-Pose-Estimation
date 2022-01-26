import cv2
import mediapipe as mp
import numpy as np
from time import sleep
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def isStanding(Lhip, Rhip, Lknee, Rknee):
    # Returns 1 - Standing; 0 - Not Standing
    # Check for Standing: hip & knee within certain horizontal distance
    # Do this for both L and R
    epsilon_stand = 20
    if np.abs(Lhip[0] - Lknee[0])<epsilon_stand and np.abs(Rhip[0] - Rknee[0]) < epsilon_stand:
        return 1
    return 0


def isLyingDown(Lshoulder, Rshoulder, Lhip, Rhip):
    # Returns 1 - Lying Down; 0 - Not Lying Down
    # Test for Lying Down: shoulder & hip within certain vertical distance
    # Do this for both L and R
    epsilon_lying = 50
    if np.abs(Lhip[1] - Lshoulder[1]) < epsilon_lying and np.abs(Rhip[1] - Rshoulder[1]) < epsilon_lying:
        return 1
    return 0


def isSitting(Lknee, Rknee, Lhip, Rhip, Lshoulder, Rshoulder):
    # Returns 0: Sitting; 1 - Not Sitting
    # Test for sitting: angle between knee, hip, shoulder close to 90 degrees AND knee, hip within certain vertical distance
    # Do for L and R

    angleL = calculate_angle(Lknee, Lhip, Lshoulder)
    angleR = calculate_angle(Rknee, Rhip, Rshoulder)
    epsilon_sitting = 50

    if not (80 < angleL < 150 or 80 < angleR < 150):
        return 0
    if np.abs(Lhip[1] - Lknee[1]) < epsilon_sitting and np.abs(Rhip[1] - Rknee[1]) < epsilon_sitting:
        return 1
    return 0


def hasFallen(states):
    # Called ONLY when current state is 0 (lying down)
    # states: array with at most 10 elements
    # Test for Fallen: Go from standing (1) to lying down (0) within
    # 10 most recent states (i.e., check through states for element 1)
    # Note that a linear scan is sufficient as len(states) < 6 (small)
    for element in states:
        if element == 1 or element == 3:
            return 1
    return 0


def obtain_state(Lknee, Rknee, Lhip, Rhip, Lshoulder, Rshoulder, states):
    # Returns 0 - Lying Down; 1 - Standing Up; 2 - Sitting Down; 3 - Fallen; 4 - None of these (N/A)
    if isLyingDown(Lshoulder, Rshoulder, Lhip, Rhip):
        if hasFallen(states):
            return 3
        return 0
    if isStanding(Lhip, Rhip, Lknee, Rknee):
        return 1
    if isSitting(Lknee, Rknee, Lhip, Rhip, Lshoulder, Rshoulder):
        return 2
    return 4


cap = cv2.VideoCapture(0)

# Variables required
state = None
prev_states = []
status = {0: "Lying Down",
          1: "Standing Up",
          2: "Sitting Down",
          3: "Fallen",
          4: "N/A",
          None: "Waiting..."
          }

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        image_height, image_width, _ = frame.shape

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            Lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            Rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            Lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            Rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            Lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            Rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            Lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            Rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            # Calculate angles
            Langle1 = calculate_angle(Lhip, Lknee, Lankle)
            Rangle1 = calculate_angle(Rhip, Rknee, Rankle)

            Langle2 = calculate_angle(Lknee, Lhip, Lshoulder)
            Rangle2 = calculate_angle(Rknee, Rhip, Rshoulder)

            # Scaling of co-ordinates for function
            Lhip[0] *= image_width
            Lhip[1] *= image_height
            Lknee[0] *= image_width
            Lknee[1] *= image_height
            Lshoulder[0] *= image_width
            Lshoulder[1] *= image_height

            Rhip[0] *= image_width
            Rhip[1] *= image_height
            Rknee[0] *= image_width
            Rknee[1] *= image_height
            Rshoulder[0] *= image_width
            Rshoulder[1] *= image_height

            # Ensures prev_states stays within 10 elements
            if len(prev_states) > 5:
                while len(prev_states) > 5:
                    prev_states.pop(0)

            # Get current state and add to prev_states array
            state = obtain_state(Lknee, Rknee, Lhip, Rhip, Lshoulder, Rshoulder, prev_states)
            prev_states.append(state)


        except:
            pass

        # Render bed exit system
        # Setup status box
        cv2.rectangle(image, (0, 0), (500, 73), (245, 117, 16), -1)

        # Stage data
        cv2.putText(image, 'STATE: ', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(status[state]),
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()