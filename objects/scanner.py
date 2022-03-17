import cv2 as cv
import document_scanner as ds


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
                processed = ds.document_selection_preprocessing(frame)

                contours, h = cv.findContours(
                    processed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
                )

                frame_contours = cv.drawContours(
                    frame.copy(), contours, -1, (0, 0, 200), 5
                )

                try:
                    b = max(contours, key=cv.contourArea)
                    # print(len(b))
                    frame_contours = cv.drawContours(
                        frame_contours, [b], -1, (0, 200, 0), 5
                    )
                    b = ds.approxContours(b, epsilon=0.1 / 10)
                    frame_contours = cv.drawContours(
                        frame_contours, b, -1, (200, 0, 0), 10
                    )
                    if len(b) == 4:
                        document = ds.alignSelection(frame.copy(), b)
                        cv.imshow("document", document)

                except Exception as e:
                    print(e)

                cv.imshow("main", frame_contours)

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
    ip_cam_url = getIPCamUrl()
    main()
    # main(ip_cam_url)
