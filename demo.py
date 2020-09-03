from face_detector import FaceDetector
from face_landmark import FaceLandmark
import cv2


def camera_run():
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        detec_image = image.copy()
        landmark_image = image.copy()
        detections, _ = face_detector_handle.run(detec_image)
        for detection in detections:
            cv2.rectangle(image, (int(detection[0]), int(detection[1])),
                          (int(detection[2]), int(detection[3])), (0, 0, 255), 2)
            cv2.putText(image, 'face:' + str(detection[4]), (int(detection[0]), int(detection[1]) - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
            landmarks, states = face_landmark_handle.run(landmark_image, detection)
            for point in landmarks:
                cv2.circle(image, center=(int(point[0]), int(point[1])),
                           color=(255, 122, 122), radius=1, thickness=1)
        cv2.imshow('', image)
        if cv2.waitKey(1) in (27, 32):  # space esc
            break


def image_run():
    image = cv2.imread('data/1.jpeg')
    detec_image = image.copy()
    landmark_image = image.copy()
    detections, _ = face_detector_handle.run(detec_image)
    for detection in detections:
        cv2.rectangle(image, (int(detection[0]), int(detection[1])),
                      (int(detection[2]), int(detection[3])), (0, 0, 255), 2)
        cv2.putText(image, 'face:' + str(detection[4]), (int(detection[0]), int(detection[1]) - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        landmarks, states = face_landmark_handle.run(landmark_image, detection)
        for point in landmarks:
            cv2.circle(image, center=(int(point[0]), int(point[1])),
                       color=(255, 122, 122), radius=1, thickness=1)
    cv2.imshow('', image)
    cv2.waitKey()


if __name__ == '__main__':
    face_detector_handle = FaceDetector(top_k=10)
    face_landmark_handle = FaceLandmark()
    # camera_run()
    image_run()