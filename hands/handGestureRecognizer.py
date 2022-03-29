import cv2 as cv
import numpy as np
import tensorflow as tf

from handProcessor import HandProcessor

handProcessor = HandProcessor(max_num_hands=1)


model = tf.keras.models.load_model("hands/hand_gesture_model")
print(model.summary())

gesture_types = ["open", "close", "point", "v", "horns", "three"]


def normalize_hand(hand: np.ndarray):
    arr = hand - np.min(hand, axis=0)
    return arr / np.max(arr, axis=0)


def perdict_gesture(hand):
    # n = np.array([hand - np.min(hand, axis=0)])
    n = np.array([normalize_hand(hand)])
    res = model(n)
    max_arg = np.argmax(res, axis=1)[0]

    current_gesture = gesture_types[max_arg]
    probability = int(res[0, max_arg] * 100)
    return current_gesture, probability


def write_text(frame, org, text, fg=(200, 200, 200), bg=(80, 0, 80)):
    x, y = org
    cv.putText(
        frame,
        text,
        (x, y - 20),
        cv.FONT_HERSHEY_PLAIN,
        fontScale=3,
        color=bg,
        thickness=6,
    )
    cv.putText(
        frame,
        text,
        (x, y - 20),
        cv.FONT_HERSHEY_PLAIN,
        fontScale=3,
        color=fg,
        thickness=2,
    )


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
                h, w, c = frame.shape
                handProcessor.proc(frame)

                hands = handProcessor.get_landmark_as_np_arr()
                for hand in hands:
                    for (x1, y1) in hand:
                        pt1 = (int(x1 * w), int(y1 * h))
                        cv.circle(frame, pt1, 5, (200, 100, 0), -1)

                    current_gesture, probability = perdict_gesture(hand)

                    pt = hand[4]
                    (x, y) = (int(pt[0] * w), int(pt[1] * h))
                    write_text(frame, (x, y), f"{current_gesture} %{probability}")

                cv.imshow("main", frame)

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


def getIPCamUrl():
    usr, pas = "aaa", "aaa"
    ip = "192.168.50.175"
    port = "8080"
    return f"http://{usr}:{pas}@{ip}:{port}/video"


if __name__ == "__main__":
    ip_cam = getIPCamUrl()
    main(ip_cam)
    # main()
