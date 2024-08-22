import cv2 
import mediapipe as mp
import time
import handtrackingmodule as htm

ptime=0
ctime=0
cap = cv2.VideoCapture(0)
detector = htm.HandDetector()
while True:
    success, img = cap.read()
    if not success or img is None:
        print("Warning: Failed to capture image. Skipping frame.")
        continue

    img, results = detector.findHands(img)
    lmlist = detector.findPosition(img, results)
    if len(lmlist)!= 0:
        print(lmlist[4])

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
