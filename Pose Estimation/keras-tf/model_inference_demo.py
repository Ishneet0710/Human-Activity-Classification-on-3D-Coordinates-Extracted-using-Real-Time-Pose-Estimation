from IPython.core.events import post_execute
import tensorflow as tf 
from tensorflow import keras
import cv2 
import mediapipe as mp
import pandas as pd
import numpy as np

def point():
        x = [
            'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22',
            'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33',
        ]
        y = [
            'y12', 'y13', 'y14', 'y15', 'y16', 'y17', 'y18', 'y19', 'y20', 'y21', 'y22',
            'y23', 'y24', 'y25', 'y26', 'y27', 'y28', 'y29', 'y30', 'y31', 'y32', 'y33',
        ]
        z = [
            'z12', 'z13', 'z14', 'z15', 'z16', 'z17', 'z18', 'z19', 'z20', 'z21', 'z22',
            'z23', 'z24', 'z25', 'z26', 'z27', 'z28', 'z29', 'z30', 'z31', 'z32', 'z33',
        ]
        coords = [x, y, z]
        return coords


def display_tflite_classify_pose(cap, model):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False    

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Pose Detections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Concate rows
                    row = pose_row

                    # point[11] - point[32] input to tflite model.
                    coords = point()
                    specify_float = 8

                    dict_p12_to_p33 = {
                        # x12 to x33
                        coords[0][0]:round(row[44], specify_float),
                        coords[0][1]:round(row[48], specify_float),
                        coords[0][2]:round(row[52], specify_float),
                        coords[0][3]:round(row[56], specify_float),
                        coords[0][4]:round(row[60], specify_float),
                        coords[0][5]:round(row[64], specify_float),
                        coords[0][6]:round(row[68], specify_float),
                        coords[0][7]:round(row[72], specify_float),
                        coords[0][8]:round(row[76], specify_float),
                        coords[0][9]:round(row[80], specify_float),
                        coords[0][10]:round(row[84], specify_float),
                        coords[0][11]:round(row[88], specify_float),
                        coords[0][12]:round(row[92], specify_float),
                        coords[0][13]:round(row[96], specify_float),
                        coords[0][14]:round(row[100], specify_float),
                        coords[0][15]:round(row[104], specify_float),
                        coords[0][16]:round(row[108], specify_float),
                        coords[0][17]:round(row[112], specify_float),
                        coords[0][18]:round(row[116], specify_float),
                        coords[0][19]:round(row[120], specify_float),
                        coords[0][20]:round(row[124], specify_float),
                        coords[0][21]:round(row[128], specify_float),

                        # y12 to y33
                        coords[1][0]:round(row[45], specify_float),
                        coords[1][1]:round(row[49], specify_float),
                        coords[1][2]:round(row[53], specify_float),
                        coords[1][3]:round(row[57], specify_float),
                        coords[1][4]:round(row[61], specify_float),
                        coords[1][5]:round(row[65], specify_float),
                        coords[1][6]:round(row[69], specify_float),
                        coords[1][7]:round(row[73], specify_float),
                        coords[1][8]:round(row[77], specify_float),
                        coords[1][9]:round(row[81], specify_float),
                        coords[1][10]:round(row[85], specify_float),
                        coords[1][11]:round(row[89], specify_float),
                        coords[1][12]:round(row[93], specify_float),
                        coords[1][13]:round(row[97], specify_float),
                        coords[1][14]:round(row[101], specify_float),
                        coords[1][15]:round(row[105], specify_float),
                        coords[1][16]:round(row[109], specify_float),
                        coords[1][17]:round(row[113], specify_float),
                        coords[1][18]:round(row[117], specify_float),
                        coords[1][19]:round(row[121], specify_float),
                        coords[1][20]:round(row[125], specify_float),
                        coords[1][21]:round(row[129], specify_float),

                        # z12 to z33
                        coords[2][0]:round(row[46], specify_float),
                        coords[2][1]:round(row[50], specify_float),
                        coords[2][2]:round(row[54], specify_float),
                        coords[2][3]:round(row[58], specify_float),
                        coords[2][4]:round(row[62], specify_float),
                        coords[2][5]:round(row[66], specify_float),
                        coords[2][6]:round(row[70], specify_float),
                        coords[2][7]:round(row[74], specify_float),
                        coords[2][8]:round(row[78], specify_float),
                        coords[2][9]:round(row[82], specify_float),
                        coords[2][10]:round(row[86], specify_float),
                        coords[2][11]:round(row[90], specify_float),
                        coords[2][12]:round(row[94], specify_float),
                        coords[2][13]:round(row[98], specify_float),
                        coords[2][14]:round(row[102], specify_float),
                        coords[2][15]:round(row[106], specify_float),
                        coords[2][16]:round(row[110], specify_float),
                        coords[2][17]:round(row[114], specify_float),
                        coords[2][18]:round(row[118], specify_float),
                        coords[2][19]:round(row[122], specify_float),
                        coords[2][20]:round(row[126], specify_float),
                        coords[2][21]:round(row[130], specify_float),

                    }
                    input_dict = {name: np.expand_dims(np.array(value, dtype=np.float32), axis=0) for name, value in dict_p12_to_p33.items()}
                    

                    result = tflite_inference(input=input_dict, model=model)
                    body_language_class = np.argmax(result)
                    body_language_prob = result[np.argmax(result)]

                    if str(body_language_class) == '0':
                        pose_class = 'Standing' 
                        print('standing')
                    elif str(body_language_class) == '1':
                        pose_class = 'Sitting'
                        print('sitting')
                    else:
                        pose_class = 'Lying'
                        print('lying')
                    
                    print(f'class: {body_language_class}, prob: {body_language_prob}')

                    # Show pose category near the ear.
                    coords = tuple(np.multiply(
                        np.array(
                            (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                            results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                        [1280,480]
                    ).astype(int))

                    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+200, coords[1]-30), 
                                (245, 117, 16), -1)
                    cv2.putText(image, pose_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Get status box
                    cv2.rectangle(image, (10,0), (310, 55), (0, 0, 0), -1)

                    # Display Class
                    cv2.putText(
                        image, 
                        'CLASS: ', (15, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 255, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, 
                        pose_class, (120, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    # Display Probability
                    cv2.putText(
                        image, 
                        'PROB: ', (15, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 255, 0), 1, cv2.LINE_AA
                    )
                    cv2.putText(
                        image, 
                        str(body_language_prob), (120, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (255, 255, 255), 2, cv2.LINE_AA
                    )

                except:
                    pass

                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            else:
                break

    print('Done!')
    cap.release()
    cv2.destroyAllWindows()


def tflite_inference(input, model):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Verify the TensorFlow Lite model. 
    for i, (name, value) in enumerate(input.items()):

        input_value = np.expand_dims(value, axis=1)
        interpreter.set_tensor(input_details[i]['index'], input_value)
        interpreter.invoke()
    output = interpreter.tensor(output_details[0]['index'])()[0]

    return output



if __name__ == '__main__':

    # 0: standing, 1: sitting, 2: lying
    video_file_name = 'Video'
    video_path = './video/'+ video_file_name + '.mp4'
    
    tflite_model = './tflite_model/model.tflite'

    cap = cv2.VideoCapture(video_path)


    display_tflite_classify_pose(cap, model=tflite_model)
