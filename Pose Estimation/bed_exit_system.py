import cv2
import mediapipe as mp
import numpy as np
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


def standing_or_lying_down(Lhip, Rhip, Lknee, Rknee):
    # Given angle close enough to 180 degrees (Logic later on)
    # returns 1 if standing, 0 if lying and 2 if neither (i.e., Michael Jackson lol)

    # Check for standing: hip & knee within certain horizontal distance
    # Do this for both L and R

    # Check for lying down: hip & knee within certain vertical distance
    # Do this for both L and R
    epsilon_stand = 20
    epsilon_lying = 50

    # Test for Standing
    if np.abs(Lhip[0] - Lknee[0])<epsilon_stand and np.abs(Rhip[0] - Rknee[0]) < epsilon_stand:
        return 1

    # Test for Lying Down
    if np.abs(Lhip[1] - Lknee[1]) < epsilon_lying and np.abs(Rhip[1] - Rknee[1]) < epsilon_lying:
        return 0

    # If neither standing nor lying down return 2
    return 2


cap = cv2.VideoCapture(0)

# Variables required
state = None # 1 for standing, 0 for lying down, 2 for n/a
status = {0: "Lying Down",
          1: "Standing Up",
          2: "N/A"
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

            Rhip[0] *= image_width
            Rhip[1] *= image_height
            Rknee[0] *= image_width
            Rknee[1] *= image_height

            # Check if both angle is close enough to 180 degree
            if Langle1 > 160 and Rangle1 > 160 and Langle2 > 160 and Rangle2 > 160:
                state = standing_or_lying_down(Lhip, Rhip, Lknee, Rknee)
            else:
                state = 2


            # # Visualize angle
            # cv2.putText(image, str(Rangle),
            #             tuple(np.multiply(Rknee, [640, 480]).astype(int)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            #             )
            #debugging and printing of coordinates
            # print("Left Hip x: ", Lhip[0], "\nLeft Hip y: ", Lhip[1])
            # print("\nLeft Knee x: ", Lknee[0], "\nLeft Knee y: ", Lknee[1])
            # print("\nRight Hip x: ", Rhip[0], "\nRight Hip y: ", Rhip[1])
            # print("\nRight Knee x: ", Rknee[0], "\nRight Knee y: ", Rknee[1])
            # print("\n")
            # if standing_or_lying_down(Lhip, Rhip, Lknee, Rknee) == 2:
            #     print("N/A")
            # elif standing_or_lying_down(Lhip, Rhip, Lknee, Rknee):
            #     print("Standing")
            # elif standing_or_lying_down(Lhip, Rhip, Lknee, Rknee) == 0:
            #     print("Lying Down")

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