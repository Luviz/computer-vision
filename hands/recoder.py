import cv2 as cv
import numpy as np
from os import makedirs
from glob import glob

from handProcessor import HandProcessor

handProcessor = HandProcessor(min_detection_confidence=0.7, max_num_hands=1)

rootFolder = "hands/"

gesture_types = ["open", "close", "point", "v", "horns", "three"]
current_gesture = 0


def create_folders():
    for gesture in gesture_types:
        try:
            makedirs(f"{rootFolder}/gestures/{gesture}")
        except FileExistsError:
            ...


def normalize_hand(hand: np.ndarray):
    arr = hand - np.min(hand, axis=0)
    return arr / np.max(arr, axis=0)


def take(hands: np.ndarray, folder_path=rootFolder):
    for hand in hands:
        values = normalize_hand(hand)
        filename = str(len(glob(folder_path + "*"))) + ".txt"
        filepath = f"{folder_path}{filename}"
        np.savetxt(filepath, values)


def main(cam_src=None):
    global current_gesture

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
                h, w, c = frame.shape
                handProcessor.proc(frame)
                hands = handProcessor.get_landmark_as_np_arr()

                cv.putText(
                    frame,
                    gesture_types[current_gesture],
                    (0, h - 20),
                    cv.FONT_HERSHEY_PLAIN,
                    fontScale=4,
                    color=(200, 0, 200),
                    thickness=4,
                )

                for hand in hands:
                    for (x1, y1) in hand:
                        pt1 = (int(x1 * w), int(y1 * h))
                        cv.circle(frame, pt1, 5, (200, 100, 0), -1)

                cv.imshow("main", frame)
            waitKey = cv.waitKey(10)
            if waitKey > 0:
                # print(waitKey, chr(waitKey))
                if waitKey in [ord(str(i)) for i in range(10)]:
                    print(f"{waitKey=}, {key}")
                    key = int(chr(waitKey))
                if waitKey == ord("q") or waitKey == 27:
                    run = False
                if waitKey == ord(" "):
                    print("take")
                    take(
                        hands,
                        f"{rootFolder}/gestures/{gesture_types[current_gesture]}/",
                    )
                if waitKey == 81 or waitKey == 83:
                    diff = waitKey - 82
                    current_gesture = (current_gesture + diff) % len(gesture_types)
                    print(f"{current_gesture}, {gesture_types[current_gesture]}")

    except KeyboardInterrupt as e:
        print("quiting")


if __name__ == "__main__":
    create_folders()
    main()
