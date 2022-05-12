import math
import cv2 as cv
import mediapipe.python.solutions.face_detection as mpFace
import mediapipe.python.solutions.drawing_utils as mpdUtils
import mediapipe.python.solutions.drawing_styles as mpdStyles
from numpy import ndarray

from faceProtocol import FaceProtocol

faceDetection = mpFace.FaceDetection()

styles = mpdStyles.DrawingSpec(
    circle_radius=1,
)


def main(cam_src=None):
    if cam_src == None:
        cam_src = 0
    cap = cv.VideoCapture(cam_src)

    c = 0
    key = 0
    run = True
    _, last_frame = cap.read()
    try:
        while run:
            c = c + 1
            has_frame, frame = cap.read()

            if has_frame:
                h, w, _ = frame.shape
                faces = faceDetection.process(frame[:, :, ::-1])
                if faces and faces.detections:
                    for face in faces.detections:
                        drawFace(frame, face)
                        mpdUtils.draw_detection(frame, face)

                ## last frame
                last_frame = frame.copy()

            cv.imshow("main", frame)

            waitKey = cv.waitKey(1)
            if waitKey > 0:
                # print(waitKey, chr(waitKey))
                if waitKey in [ord(str(i)) for i in range(10)]:
                    print(f"{waitKey=}, {key}")
                    key = int(chr(waitKey))
                # if waitKey == ord(1):
                #     pass
                if waitKey == ord("q") or waitKey == 27:
                    run = False

    except KeyboardInterrupt as e:
        print("quiting")


def drawFace(image: ndarray, face: FaceProtocol, color=(100, 200, 0)):
    h, w, _ = image.shape
    faceRBB = face.location_data.relative_bounding_box
    score = int(face.score[0] * 100)
    pt1 = (int(faceRBB.xmin * w) - 10, int(faceRBB.ymin * h) - 10)
    pt2 = (
        int((faceRBB.xmin + faceRBB.width) * w) + 10,
        int((faceRBB.ymin + faceRBB.height) * h) + 10,
    )

    ## Draw bonding box
    cv.rectangle(image, pt1, pt2, color, thickness=2, lineType=cv.LINE_8)

    ## Draw score y min - 10
    org = (pt1[0], pt1[1] - 10)
    cv.putText(
        image,
        str(score),
        org,
        color=color,
        fontFace=cv.FONT_HERSHEY_DUPLEX,
        fontScale=1,
        thickness=2,
    )


if __name__ == "__main__":
    usr = pas = "aaa"
    ip = "192.168.50.175"
    port = "8080"
    ip_cam_url = f"http://{usr}:{pas}@{ip}:{port}/video"
    # main()
    main(ip_cam_url)
