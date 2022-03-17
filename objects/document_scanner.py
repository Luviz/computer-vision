import cv2 as cv
import numpy as np


def approxContours(cntr, epsilon=0.1):
    peri = cv.arcLength(cntr, True)
    return cv.approxPolyDP(cntr, epsilon * peri, True)


def distance(p1, p2) -> int:
    s1, s2 = np.subtract(p1, p2)
    return round(np.sqrt(s1**2 + s2**2))


def getCenter(cntr):
    m = cv.moments(cntr)
    return int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])


def getCornerPoints(cnt):
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return {"l": leftmost, "r": rightmost, "t": topmost, "b": bottommost}


def alignSelection(img, cntr):
    points = getCornerPoints(cntr)
    topleft = points["l"]
    width = distance(topleft, points["t"])
    height = distance(topleft, points["b"])

    to_points = np.float32([(0, 0), (width, 0), (0, height), (width, height)])
    from_points = np.float32([points["l"], points["t"], points["b"], points["r"]])

    M = cv.getPerspectiveTransform(from_points, to_points)
    return cv.warpPerspective(img, M, (width, height))
