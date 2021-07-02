import cv2
import mediapipe as mp
import time


class FaceDetector():

    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            self.minDetectionCon)

    def findFaces(self, img, draw=True):

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(rgb_img)
        bboxes = []

        if self.results.detections:
            for i, detection in enumerate(self.results.detections):
                # To draw boundry box and landmarks using mpDraw
                # mpDraw.draw_detection(img, detection)
                h, w, c = img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin *
                                                h), int(bboxC.width * w), int(bboxC.height * h)
                bboxes.append([i, bbox, detection.score])

                if draw:
                    self.fancyDraw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

        return img, bboxes

    def fancyDraw(self, img, bbox, l=30, t=3, rt=1):

        x, y, width, height = bbox
        x1, y1 = x + width, y + height

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)
        # Top Right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
        # Bottom Left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img


def main():
    cap = cv2.VideoCapture(0)
    prevTime = 0
    detector = FaceDetector()

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


if __name__ == "__main__":
    main()
