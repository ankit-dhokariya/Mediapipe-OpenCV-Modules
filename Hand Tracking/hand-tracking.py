import cv2
import time
import HandTrackingModule as htmod

currTime = 0
prevTime = 0

cap = cv2.VideoCapture(0)

detector = htmod.handDetector()

while True:
    _, img = cap.read()

    img = detector.findHands(img)
    landmarksList = detector.findPosition(img, draw=False)

    if len(landmarksList) != 0:
        print(landmarksList[4])

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)
