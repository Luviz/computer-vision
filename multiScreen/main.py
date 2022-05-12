import time
import cv2 as cv
import numpy as np


def main(cam_src=None):
    if cam_src == None:
        cam_src = 0
    cap = cv.VideoCapture(cam_src)
    c = 0
    key = 0
    run = True
    fps_lim = 25.0
    cap.set(cv.CAP_PROP_FPS, fps_lim)
    current_frame_time = time.time()
    previous_frame_time = time.time()
    out = cv.VideoWriter(
        f"/mnt/azure_storage/azblob/recordings/{int(current_frame_time)}.mp4",
        cv.VideoWriter_fourcc(*"FMP4"),
        fps_lim,
        (852, 160),
    )
    try:
        while run:
            c = c + 1
            has_frame, frame = cap.read()
            if has_frame:
                current_frame_time = time.time()
                fps = 1 // (current_frame_time - previous_frame_time)
                h, w, _ = frame.shape
                small_frame = cv.resize(frame, (w // 3, h // 3))
                hsv = cv.cvtColor(small_frame, cv.COLOR_BGR2HSV_FULL)
                h = hsv[:, :, 0]
                s = hsv[:, :, 1]
                v = hsv[:, :, 2]

                stack = np.hstack([h, s, v])
                stack = cv.cvtColor(stack, cv.COLOR_GRAY2BGR)
                f = np.hstack([small_frame, stack])
                out.write(f)
                print(f"{fps = }")
                cv.imshow("main", f)
                previous_frame_time = current_frame_time

            waitKey = cv.waitKey(10)
            if waitKey > 0:
                # print(waitKey, chr(waitKey))
                if waitKey in [ord(str(i)) for i in range(10)]:
                    print(f"{waitKey=}, {key}")
                    key = int(chr(waitKey))
                if waitKey == ord("q") or waitKey == 27:
                    out.release()
                    run = False

    except KeyboardInterrupt as e:
        print("quiting")


if __name__ == "__main__":
    main("http://127.0.0.1:8012/video_feed")
