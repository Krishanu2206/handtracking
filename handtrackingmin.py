import cv2 
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mphands=mp.solutions.hands
hands=mphands.Hands()
mpDraw = mp.solutions.drawing_utils

ptime=0
ctime=0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLms.landmark):
                # print(id, landmark)
                h,w,c=img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                print(id, cx, cy)
                if id==4:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mphands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    