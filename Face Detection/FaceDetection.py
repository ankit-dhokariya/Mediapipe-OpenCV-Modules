import cv2
import time
import FaceDetectionModule as fdm

cap = cv2.VideoCapture(0)
prevTime = 0
detector = fdm.FaceDetector()

while True:

    success, img = cap.read()
    img, bboxes = detector.findFaces(img)
    print(bboxes)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
