import cv2
import mediapipe as mp
import math

#For hand tracking and tracing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#Initilize the video capture to webcam
vidcap = cv2.VideoCapture(0)

#perameters to window displayed
winwidth = 1280
winheight = 960

#For face tracking
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#sets color for face rectangle
rectangleColor = (0,165,255)
#boolean paremeter for knowing if the face identified will be blured or not
blur = False

#for delaying the proccess of bluring or unbluring the face
time = 0
pressed = False

with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        if not ret:
            break

        #face capturing
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

            #blur face 
            if blur:
                roi = frame[y:y+h, x:x+w] 
                roi = cv2.GaussianBlur(roi, (23, 23), 30) 
                frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

                cv2.putText(frame, "blurred", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        #hand captured
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        process_frames = hands.process(rgb_frame)
        img_h, img_w, _= frame.shape

        if process_frames.multi_hand_landmarks:
            for lm in process_frames.multi_hand_landmarks:
                if time == 0:
                    #blurs and unblurs face detected based on if the tips of the thumb and middle finger touch together
                    x1, y1 = int(lm.landmark[mp_hands.HandLandmark.THUMB_TIP].x*img_w), int(lm.landmark[mp_hands.HandLandmark.THUMB_TIP].y*img_h)
                    x2, y2 = int(lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x*img_w), int(lm.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y*img_h)
                    length = math.hypot(x2-x1,y2-y1)

                    if not pressed and length < 20:
                        blur = True if not blur else False
                        pressed = True
                    elif pressed and length > 20:
                        pressed = False
                        time = 18
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        if time > 0:
            time -= 1

        resized_frame = cv2.resize(frame, (winwidth, winheight))

        cv2.imshow('Hand Detected Blur', resized_frame)

        #if key 1 is pressed then ends the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

vidcap.release()
cv2.destroyAllWindows()