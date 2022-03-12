import math
import cv2 as cv
import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpdUtils

hands = mpHands.Hands()


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
                processed_hands = hands.process(frame[:, :, ::-1])
                if processed_hands.multi_hand_landmarks:
                    for lm in processed_hands.multi_hand_landmarks:
                        color = (255, 0, 0)
                        (width, height, _) = frame.shape
                        dist = getLandMarkDistance(
                            lm.landmark[4], lm.landmark[8], width, height
                        )
                        cv.line(
                            frame,
                            pt1=landmarkCoored(lm.landmark[4], width, height),
                            pt2=landmarkCoored(lm.landmark[8], width, height),
                            color=(100, 255, 250),
                            thickness=2,
                        )

                        cv.putText(
                            img=frame,
                            text=f"{round(dist, 2)}",
                            org=landmarkCoored(lm.landmark[4], width, height),
                            fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            color=(110, 110, 110),
                            fontScale=1,
                            thickness=4,
                        )

                        cv.putText(
                            img=frame,
                            text=f"{round(dist, 2)}",
                            org=landmarkCoored(lm.landmark[4], width, height),
                            fontFace=cv.FONT_HERSHEY_SIMPLEX,
                            color=(0, 200, 0),
                            fontScale=1,
                            thickness=1,
                        )

                        drawLandmark(frame, width, height, lm.landmark[4], (255, 0, 0))
                        drawLandmark(frame, width, height, lm.landmark[8], (0, 255, 0))
                        drawLandmark(frame, width, height, lm.landmark[12], (0, 0, 255))
                        drawLandmark(
                            frame,
                            width,
                            height,
                            lm.landmark[16],
                            (255, 0, 255),
                            radius=3,
                        )
                        drawLandmark(
                            frame,
                            width,
                            height,
                            lm.landmark[20],
                            (255, 255, 0),
                            radius=3,
                        )

                        # mpdUtils.draw_landmarks(
                        #     image=frame,
                        #     landmark_list=lm,
                        #     landmark_drawing_spec=None,
                        #     connections=mpHands.HAND_CONNECTIONS,
                        # )

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


def landmarkCoored(landmark, width, height):
    return (int(landmark.x * height), int(landmark.y * width))


def drawLandmark(frame, width, height, landmark, color, radius=8, thickness=4):
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
