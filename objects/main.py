import cv2 as cv
import numpy as np


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
                work_img = preprocess(frame)

                contours, _ = cv.findContours(
                    work_img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE
                )

                contours = sorted(contours, key=cv.contourArea, reverse=True)
                approx_contours = [approxContours(c, 0.01) for c in contours]
                approx_contours = sorted(approx_contours, key=len)

                rectangles = []
                for cntr in approx_contours:
                    if len(cntr) > 16:
                        break
                    if len(cntr) == 4:
                        rectangles.append(cntr)
                    cv.drawContours(frame, [cntr], -1, (0, 255, 0), 3)

                if rectangles:
                    largest = max(rectangles, key=cv.contourArea)
                    cv.drawContours(frame, [largest], -1, (0, 0, 200), 3)

                cv.imshow("work_img", work_img)
                cv.imshow("main", frame)

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


def preprocess(frame):
    kernel = np.ones((5, 5))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (81, 81), 1)
    ret, thresh = cv.threshold(blur, 130, 255, cv.THRESH_BINARY)
    canny = cv.Canny(thresh, 0, 255)
    dilate = cv.dilate(canny, kernel, iterations=5)
    erode = cv.erode(dilate, kernel, iterations=5)
    return erode


def approxContours(cntr, epsilon=0.1):
    peri = cv.arcLength(cntr, True)
    return cv.approxPolyDP(cntr, epsilon * peri, True)


if __name__ == "__main__":
    usr = pas = "aaa"
    # ip = "192.168.50.175"
    ip = "192.168.251.187"
    port = "8080"
    ip_cam_url = f"http://{usr}:{pas}@{ip}:{port}/video"
    main(0)
    # main(ip_cam_url)
