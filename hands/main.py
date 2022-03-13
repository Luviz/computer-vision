import math
import cv2 as cv
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpdUtils

# from hands. import HandProcessor
from handProtocol import Processed_hands
from handProcessor import HandData, HandProcessor


hands = HandProcessor(min_detection_confidence=0.7)


def main(cam_src=None):
    if cam_src == None:
        cam_src = 0
    cap = cv.VideoCapture(cam_src)

    c = 0
    key = 0
    run = True
    _, last_frame = cap.read()

    buttonArray = [draw_button((10, i + 10), (80, i + 40)) for i in range(0, 121, 40)]

    try:
        while run:
            has_frame, frame = cap.read()

            if has_frame:
                frame = cv.flip(frame, 1)
                w, h, c = frame.shape
                processed_hands = hands.proc(frame[:, :, ::-1])

                if processed_hands.multi_hand_landmarks:
                    # drawHands(processed_hands)
                    hd = hands.get_handData()

                    for data in hd:
                        # drawHands(frame, data)
                        index_finger = landmarkCoored(data.landmarks[8], w, h)

                        for btn in buttonArray:
                            btn(frame, index_finger)

                else:
                    for btn in buttonArray:
                        btn(frame, (0, 0))

                cv.imshow("main", frame)

                ## last frame
                last_frame = frame.copy()

            waitKey = cv.waitKey(1)
            if waitKey > 0:
                # print(waitKey, chr(waitKey))
                if waitKey in [ord(str(i)) for i in range(10)]:
                    print(f"{waitKey=}, {key}")
                    key = int(chr(waitKey))
                if waitKey == ord("q") or waitKey == 27:
                    run = False

    except KeyboardInterrupt as e:
        print("quiting")


def inBound(pt, box):
    return (box["pt1"][0] < pt[0] < box["pt2"][0]) and (
        box["pt1"][1] < pt[1] < box["pt2"][1]
    )


def draw_button(pt1, pt2, color=(220, 220, 220), color_a=(126, 225, 126)):
    def draw(image, point):
        active = inBound(point, {"pt1": pt1, "pt2": pt2})
        cv.rectangle(image, pt1, pt2, color=(color_a if active else color), thickness=3)

    return draw


def draw_distance(image, lm1, lm2):
    (w, h, c) = image.shape
    color1 = (200, 0, 0)
    color2 = (0, 200, 0)

    dist = getLandMarkDistance(lm1, lm2, w, h)
    pt1 = landmarkCoored(lm1, w, h)
    pt2 = landmarkCoored(lm2, w, h)

    drawText(image, str(int(dist)), pt1, offsetText=20)

    cv.line(image, pt1, pt2, color=(100, 255, 250), thickness=2)

    drawLandmark(image, lm1, color1)
    drawLandmark(image, lm2, color2)


def drawText(
    img, text, org, fg_color=(0, 0, 0), bg_color=(170, 170, 170), offsetText=None
):
    if offsetText != None:
        org = (org[0] + offsetText, org[1] - offsetText)

    cv.putText(  # textout line
        img,
        text,
        org,
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        color=bg_color,
        fontScale=1,
        thickness=5,
    )

    cv.putText(
        img,
        text,
        org,
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        color=fg_color,
        fontScale=1,
        thickness=2,
    )


def drawHands(frame, processed_hands: HandData):
    try:
        w, h, _ = frame.shape
        label = processed_hands.classification.label
        lm = processed_hands.landmarks

        mpdUtils.draw_landmarks(
            image=frame,
            landmark_list=processed_hands.multi_hand_landmarks,
            landmark_drawing_spec=None,
            connections=mpHands.HAND_CONNECTIONS,
        )

        drawLandmark(frame, lm[4], (0, 100, 200), thickness=-1)
        drawLandmark(frame, lm[8], (200, 100, 0), thickness=-1)

        drawText(
            frame,
            label,
            landmarkCoored(lm[0], w, h),
            bg_color=(0, 0, 200),
        )

    except Exception as e:
        print("boo!", type(e))


def landmarkCoored(landmark, width, height):
    return (int(landmark.x * height), int(landmark.y * width))


def drawLandmark(frame, landmark, color, radius=8, thickness=4):
    width, height, c = frame.shape
    x = landmark.x * height
    y = landmark.y * width
    cv.circle(frame, (int(x), int(y)), radius, color, thickness)


def getLandMarkDistance(lm1, lm2, width, height):
    x1, y1 = lm1.x * width, lm1.y * height
    x2, y2 = lm2.x * width, lm2.y * height

    return math.sqrt(abs(x1 - x2) ** 2 + abs(y1 - y2) ** 2)


if __name__ == "__main__":
    usr = pas = "aaa"
    ip = "192.168.50.175"
    port = "8080"
    ip_cam_url = f"http://{usr}:{pas}@{ip}:{port}/video"
    main()
    # main(ip_cam_url)
