import cv2
import time
import PoseEstimationModule as pm

cap = cv2.VideoCapture(0)
prevTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    landmarksList = detector.findPosition(img, draw=False)
    if len(landmarksList) != 0:
        print(landmarksList[14])
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, f"FPS: {int(fps)}", (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
