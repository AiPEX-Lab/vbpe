import cv2
import numpy as np
import mediapipe as mp
import os

# Initialize MediaPipe Face Mesh and Hand Tracking
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5)

MAX_FRAMES = 600

def get_face_video_size(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        count = count + 1
        if count == 20:
            break
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        # Detect faces and hands using MediaPipe
        face_results = face_mesh.process(rgb_frame)

        # Draw face landmarks and hand landmarks
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # mp_drawing.draw_landmarks(rgb_frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                # Print x, y coordinates of face landmarks
                ih, iw, _ = rgb_frame.shape
                # for landmark in face_landmarks.landmark:
                #     x, y = int(landmark.x * iw), int(landmark.y * ih)
                #     print(f"Face Landmark coordinates: x={x}, y={y}")

                mid_face_x, mid_face_y = int(face_landmarks.landmark[8].x * iw), int(face_landmarks.landmark[8].y * ih)
                
                mid_face_ref_x, mid_face_ref_y = int(face_landmarks.landmark[9].x * iw), int(face_landmarks.landmark[9].y * ih)
                
                if count == 19:
                    dist = int(np.sqrt((mid_face_ref_x - mid_face_x)**2 + (mid_face_ref_y - mid_face_y)**2))
                else:
                    dist = 1
                face_video = cv2.cvtColor(rgb_frame[mid_face_y-dist:mid_face_y+dist, mid_face_x-dist:mid_face_x+dist, :], cv2.COLOR_RGB2BGR) 
                fvh, fvw, _ = face_video.shape
                face_video_size = (fvw, fvh)
        
        if not face_results.multi_face_landmarks:
            rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ih, iw, _ = rgb_frame.shape
            face_results = face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # mp_drawing.draw_landmarks(rgb_frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                    # Print x, y coordinates of face landmarks
                    ih, iw, _ = rgb_frame.shape
                    # for landmark in face_landmarks.landmark:
                    #     x, y = int(landmark.x * iw), int(landmark.y * ih)
                    #     print(f"Face Landmark coordinates: x={x}, y={y}")

                    mid_face_x, mid_face_y = int(face_landmarks.landmark[8].x * iw), int(face_landmarks.landmark[8].y * ih)
                    
                    mid_face_ref_x, mid_face_ref_y = int(face_landmarks.landmark[9].x * iw), int(face_landmarks.landmark[9].y * ih)
                    
                    if count == 19:
                        dist = int(np.sqrt((mid_face_ref_x - mid_face_x)**2 + (mid_face_ref_y - mid_face_y)**2))
                    else: 
                        dist = 1
                    face_video = cv2.cvtColor(rgb_frame[mid_face_y-dist:mid_face_y+dist, mid_face_x-dist:mid_face_x+dist, :], cv2.COLOR_RGB2BGR) 
                    fvh, fvw, _ = face_video.shape
                    face_video_size = (fvw, fvh)
            else:
                # print('Error - Mediapipe cannot detect face')
                face_video_size = (iw, ih)

    return face_video_size

def get_hand_video_size(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        count = count + 1
        if count == 20:
            break
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and hands using MediaPipe
        face_results = face_mesh.process(rgb_frame)
        hand_results = hands.process(rgb_frame)

        if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Print x, y coordinates of face landmarks
                ih, iw, _ = rgb_frame.shape
                # for landmark in hand_landmarks.landmark:
                #     x, y = int(landmark.x * iw), int(landmark.y * ih)
                #     print(f"Hand Landmark coordinates: x={x}, y={y}")
                hand_bottom_x, hand_bottom_y = int(hand_landmarks.landmark[0].x * iw), int(hand_landmarks.landmark[0].y * ih)
                
                hand_top_x, hand_top_y = int(hand_landmarks.landmark[9].x * iw), int(hand_landmarks.landmark[9].y * ih)
                
                mid_hand_x = int((hand_bottom_x + hand_top_x)/2)
                mid_hand_y = int((hand_bottom_y + hand_top_y)/2)
                
                if count == 19:
                    dist = int(np.sqrt((mid_hand_x - hand_bottom_x)**2 + (mid_hand_y - hand_bottom_y)**2))
                else:
                    dist = 1
                hand_video = cv2.cvtColor(rgb_frame[mid_hand_y-dist:mid_hand_y+dist, mid_hand_x-dist:mid_hand_x+dist, :], cv2.COLOR_RGB2BGR)
                hvh, hvw, _ = hand_video.shape
                hand_video_size = (hvw, hvh)

        if not face_results.multi_face_landmarks:
            rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            ih, iw, _ = rgb_frame.shape
            face_results = face_mesh.process(rgb_frame)
            hand_results = hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Print x, y coordinates of face landmarks
                    ih, iw, _ = rgb_frame.shape
                    # for landmark in hand_landmarks.landmark:
                    #     x, y = int(landmark.x * iw), int(landmark.y * ih)
                    #     print(f"Hand Landmark coordinates: x={x}, y={y}")
                    hand_bottom_x, hand_bottom_y = int(hand_landmarks.landmark[0].x * iw), int(hand_landmarks.landmark[0].y * ih)
                    
                    hand_top_x, hand_top_y = int(hand_landmarks.landmark[9].x * iw), int(hand_landmarks.landmark[9].y * ih)
                    
                    mid_hand_x = int((hand_bottom_x + hand_top_x)/2)
                    mid_hand_y = int((hand_bottom_y + hand_top_y)/2)
                    
                    if count == 19:
                        dist = int(np.sqrt((mid_hand_x - hand_bottom_x)**2 + (mid_hand_y - hand_bottom_y)**2))
                    else:
                        dist = 1
                    hand_video = cv2.cvtColor(rgb_frame[mid_hand_y-dist:mid_hand_y+dist, mid_hand_x-dist:mid_hand_x+dist, :], cv2.COLOR_RGB2BGR)
                    hvh, hvw, _ = hand_video.shape
                    hand_video_size = (hvw, hvh)
            else: 
                # print('Error - Mediapipe cannot detect hand')
                hand_video_size = (iw, ih)
                
    return hand_video_size

def face_capture(video_path):        
    # Capture video from a video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
       
    size = (frame_width, frame_height)
    face_video_size = get_face_video_size(video_path)
    
    face_video_output_name = './Results/MP_Videos/Face/' + os.path.splitext(os.path.basename(video_path))[0] + '_face_output.avi'
    result_face = cv2.VideoWriter(face_video_output_name, cv2.VideoWriter_fourcc(*'MJPG'), 60, face_video_size)
    
    count = 0
    while cap.isOpened():
        count = count + 1
        if count == MAX_FRAMES:
            break
        ret, frame = cap.read()
        if not ret:
            continue
        if count >= 19:

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces and hands using MediaPipe
            face_results = face_mesh.process(rgb_frame)

            # Draw face landmarks and hand landmarks
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # mp_drawing.draw_landmarks(rgb_frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                    # Print x, y coordinates of face landmarks
                    ih, iw, _ = rgb_frame.shape
                    # for landmark in face_landmarks.landmark:
                    #     x, y = int(landmark.x * iw), int(landmark.y * ih)
                    #     print(f"Face Landmark coordinates: x={x}, y={y}")

                    mid_face_x, mid_face_y = int(face_landmarks.landmark[8].x * iw), int(face_landmarks.landmark[8].y * ih)
                    
                    mid_face_ref_x, mid_face_ref_y = int(face_landmarks.landmark[9].x * iw), int(face_landmarks.landmark[9].y * ih)
                    
                    if count == 19:
                        dist = int(np.sqrt((mid_face_ref_x - mid_face_x)**2 + (mid_face_ref_y - mid_face_y)**2))
                        
                    face_video = cv2.cvtColor(rgb_frame[mid_face_y-dist:mid_face_y+dist, mid_face_x-dist:mid_face_x+dist, :], cv2.COLOR_RGB2BGR) 
                    

            if not face_results.multi_face_landmarks:
                # print('Need to rotate all Face')
                rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                face_results = face_mesh.process(rgb_frame)
                
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        # mp_drawing.draw_landmarks(rgb_frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                        # Print x, y coordinates of face landmarks
                        ih, iw, _ = rgb_frame.shape
                        # for landmark in face_landmarks.landmark:
                        #     x, y = int(landmark.x * iw), int(landmark.y * ih)
                        #     print(f"Face Landmark coordinates: x={x}, y={y}")

                        mid_face_x, mid_face_y = int(face_landmarks.landmark[8].x * iw), int(face_landmarks.landmark[8].y * ih)
                        
                        mid_face_ref_x, mid_face_ref_y = int(face_landmarks.landmark[9].x * iw), int(face_landmarks.landmark[9].y * ih)
                        
                        if count == 19:
                            dist = int(np.sqrt((mid_face_ref_x - mid_face_x)**2 + (mid_face_ref_y - mid_face_y)**2))
                            
                        face_video = cv2.cvtColor(rgb_frame[mid_face_y-dist:mid_face_y+dist, mid_face_x-dist:mid_face_x+dist, :], cv2.COLOR_RGB2BGR) 
                             
                else:
                    # print('Error - Mediapipe cannot detect face')  
                    ih, iw, _ = rgb_frame.shape
                    face_video = cv2.cvtColor(rgb_frame[int(0.5*ih):int(0.6*ih), int(0.1*iw):int(0.4*iw), :], cv2.COLOR_RGB2BGR)

            result_face.write(face_video)


            # Break the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    result_face.release()
    
    cv2.destroyAllWindows()

def hand_capture(video_path):        
    # Capture video from a video file
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
       
    size = (frame_width, frame_height)
    
    hand_video_size = get_hand_video_size(video_path)
    hand_video_output_name = './Results/MP_Videos/Hand/' + os.path.splitext(os.path.basename(video_path))[0] + '_hand_output.avi'
    result_hand = cv2.VideoWriter(hand_video_output_name, cv2.VideoWriter_fourcc(*'MJPG'), 60, hand_video_size)
    count = 0
    while cap.isOpened():
        count = count + 1
        if count == MAX_FRAMES:
            break
        ret, frame = cap.read()
        if not ret:
            continue
        if count >= 19:

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces and hands using MediaPipe
            face_results = face_mesh.process(rgb_frame)
            hand_results = hands.process(rgb_frame)

            if hand_results.multi_hand_landmarks == True:
                print('Success')


            if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Print x, y coordinates of face landmarks
                    ih, iw, _ = rgb_frame.shape
                    # for landmark in hand_landmarks.landmark:
                    #     x, y = int(landmark.x * iw), int(landmark.y * ih)
                    #     print(f"Hand Landmark coordinates: x={x}, y={y}")
                    hand_bottom_x, hand_bottom_y = int(hand_landmarks.landmark[0].x * iw), int(hand_landmarks.landmark[0].y * ih)
                    hand_top_x, hand_top_y = int(hand_landmarks.landmark[9].x * iw), int(hand_landmarks.landmark[9].y * ih)
                    mid_hand_x = int((hand_bottom_x + hand_top_x)/2)
                    mid_hand_y = int((hand_bottom_y + hand_top_y)/2)

                    if count == 19:
                        dist = int(np.sqrt((mid_hand_x - hand_bottom_x)**2 + (mid_hand_y - hand_bottom_y)**2))
                        
                    hand_video = cv2.cvtColor(rgb_frame[mid_hand_y-dist:mid_hand_y+dist, mid_hand_x-dist:mid_hand_x+dist, :], cv2.COLOR_RGB2BGR)  
                    
            if not face_results.multi_face_landmarks:
                # print('Need to rotate all Hand')
                rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                face_results = face_mesh.process(rgb_frame)
                hand_results = hands.process(rgb_frame)
                if hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # mp_drawing.draw_landmarks(rgb_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Print x, y coordinates of face landmarks
                        ih, iw, _ = rgb_frame.shape
                        # for landmark in hand_landmarks.landmark:
                        #     x, y = int(landmark.x * iw), int(landmark.y * ih)
                        #     print(f"Hand Landmark coordinates: x={x}, y={y}")
                        hand_bottom_x, hand_bottom_y = int(hand_landmarks.landmark[0].x * iw), int(hand_landmarks.landmark[0].y * ih)
                        hand_top_x, hand_top_y = int(hand_landmarks.landmark[9].x * iw), int(hand_landmarks.landmark[9].y * ih)
                        mid_hand_x = int((hand_bottom_x + hand_top_x)/2)
                        mid_hand_y = int((hand_bottom_y + hand_top_y)/2)
                        
                        if count == 19:
                            dist = int(np.sqrt((mid_hand_x - hand_bottom_x)**2 + (mid_hand_y - hand_bottom_y)**2))
                            
                        hand_video = cv2.cvtColor(rgb_frame[mid_hand_y-dist:mid_hand_y+dist, mid_hand_x-dist:mid_hand_x+dist, :], cv2.COLOR_RGB2BGR)  
                    
                else:
                    # print('Error - Mediapipe cannot detect hand')
                    ih, iw, _ = rgb_frame.shape
                    hand_video = cv2.cvtColor(rgb_frame[int(0.5*ih):int(0.6*ih), int(0.1*iw):int(0.4*iw), :], cv2.COLOR_RGB2BGR)

            
            # Display the frame
            # cv2.imshow('Face Detection and Hand Gesture Recognition', frame)
            # result.write(frame)
            # print(face_video_size)
        
            result_hand.write(hand_video)

            # Break the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    # result_face.release()
    result_hand.release()
    cv2.destroyAllWindows()


# face_capture(video_path)
# hand_capture(video_path)
