import cv2 as cv
import numpy as np
import tensorflow as tf

import mediapipe.python.solutions.hands as mpHands
import mediapipe.python.solutions.drawing_utils as mpdUtils


from handProcessor import HandProcessor

handProcessor = HandProcessor(max_num_hands=2, min_detection_confidence=0.7)

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
        (x, y),
        cv.FONT_HERSHEY_PLAIN,
        fontScale=3,
        color=bg,
        thickness=6,
    )
    cv.putText(
        frame,
        text,
        (x, y),
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
    _, last_frame = cap.read()

    try:
        while run:
            c = c + 1
            has_frame, frame = cap.read()
            if has_frame:
                frame = cv.flip(frame, 1)
                h, w, c = frame.shape
                _ = handProcessor.proc(frame[:, :, ::-1])
                for hand in handProcessor.get_handData():
                    mpdUtils.draw_landmarks(
                        image=frame,
                        landmark_list=hand.multi_hand_landmarks,
                        landmark_drawing_spec=None,
                        connections=mpHands.HAND_CONNECTIONS,
                    )

                for hand in handProcessor.get_landmark_as_np_arr():
                    current_gesture, probability = perdict_gesture(hand)
                    center = hand.mean(axis=0)
                    pt = (center * [w, h]).astype(np.int0)
                    write_text(
                        frame, pt + [-60, 15], f"{current_gesture} %{probability}"
                    )
                    # cv.circle(frame, pt, 10, (100, 0, 0), -1)

                cv.imshow("main", frame)

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


def getIPCamUrl():
    usr, pas = "aaa", "aaa"
    ip = "192.168.50.175"
    port = "8080"
    return f"http://{usr}:{pas}@{ip}:{port}/video"


if __name__ == "__main__":
    ip_cam = getIPCamUrl()
    main(ip_cam)
    # main()
