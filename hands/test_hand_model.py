import cv2 as cv
import numpy as np

from handProcessor import HandProcessor

hands = HandProcessor(min_detection_confidence=0.7)


def main(cam_src=None):
    if cam_src == None:
        cam_src = 0
    cap = cv.VideoCapture(cam_src)

    c = 0
    key = 0
    run = True
    try:
        while run:
            c = c + 1
            has_frame, frame = cap.read()
            if has_frame:
                hands.proc(frame)
                arr = hands.get_landmark_as_np_arr()
                length = 500
                black = np.zeros((length, length))
                for hand in arr:
                    values = hand - np.min(hand, axis=0)
                    for (x1, y1), (x2, y2) in zip(hand, values):
                        pt1 = (int(x1 * length), int(y1 * length))
                        pt2 = (int(x2 * length), int(y2 * length))

                        cv.line(black, pt1, pt2, color=1, thickness=1)
                        cv.circle(black, pt1, 5, 1, -1)
                        cv.circle(black, pt2, 5, 1, 1)

                cv.imshow("main", black)

            waitKey = cv.waitKey(10)
            if waitKey > 0:
                # print(waitKey, chr(waitKey))
                if waitKey in [ord(str(i)) for i in range(10)]:
                    print(f"{waitKey=}, {key}")
                    key = int(chr(waitKey))
                if waitKey == ord("q") or waitKey == 27:
                    run = False

    except KeyboardInterrupt as e:
        print("quiting")


if __name__ == "__main__":
    main()
