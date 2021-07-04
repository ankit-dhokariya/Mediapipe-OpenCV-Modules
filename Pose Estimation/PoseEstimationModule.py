import cv2
import mediapipe as mp
import time

class poseDetector():

    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackingCon=0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode, self.upBody, self.smooth, self.detectionCon, self.trackingCon)

    def findPose(self, img, draw=True):

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(rgb_img)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):

        landmarksList = []

        if self.results.pose_landmarks:
            for i, landmark in enumerate(self.results.pose_landmarks.landmark):

                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmarksList.append([i, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return landmarksList


def main():
    cap = cv2.VideoCapture("test.mp4")
    prevTime = 0
    detector = poseDetector()

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


if __name__ == "__main__":
    main()
