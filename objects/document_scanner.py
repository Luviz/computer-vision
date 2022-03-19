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
    topleft = cntr[0][0]
    width = distance(topleft, cntr[1][0])
    height = distance(topleft, cntr[-1][0])

    to_points = np.float32([(0, 0), (0, height), (width, 0), (width, height)])
    from_points = np.float32([cntr[0][0], cntr[3][0], cntr[1][0], cntr[2][0]])

    M = cv.getPerspectiveTransform(from_points, to_points)
    return cv.warpPerspective(img, M, (width, height))


def document_selection_preprocessing(img, min_threshold=127):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    pipe = cv.GaussianBlur(gray, (3, 3), 5)
    h, pipe = cv.threshold(pipe, min_threshold, 255, cv.THRESH_BINARY)

    return pipe


def reorder_point_on_closest_distance(cntrs, pt):
    s_ix = get_smallest_index([distance(pt, cntr[0]) for cntr in cntrs])
    arr = []

    for i in range(len(cntrs)):
        ix = (i + s_ix) % len(cntrs)
        arr.append(cntrs[ix])

    return arr


def get_smallest_index(lst: list[int]) -> int:
    smallest_ix = 0
    for ix, val in enumerate(lst):
        if val < lst[smallest_ix]:
            smallest_ix = ix
    return smallest_ix
