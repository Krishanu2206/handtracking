import cv2 
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mode=mode
        self.max_num_hands=max_num_hands
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        self.mphands=mp.solutions.hands
        self.hands=self.mphands.Hands(self.mode, self.max_num_hands)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)  

        return img, results
    
    def findPosition(self, img, results, handNo=0, draw=True):
        lmlist = []
        if results and results.multi_hand_landmarks:
            requiredhand = results.multi_hand_landmarks[handNo] 
            for id, landmark in enumerate(requiredhand.landmark):
                # print(id, landmark)
                h,w,c=img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                # print(id, cx, cy)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)  

        return lmlist
    
def main():
    ptime=0
    ctime=0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
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

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
