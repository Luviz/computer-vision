import cv2 as cv


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
    ip_cam_url = getIPCamUrl()
    main()
    # main(ip_cam_url)
