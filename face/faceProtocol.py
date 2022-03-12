from typing import Any, Protocol


class RelativeBoundingBox(Protocol):
    xmin: float
    ymin: float
    width: float
    height: float


class Point(Protocol):
    x: float
    y: float


class LocationData(Protocol):
    relative_keypoints: Point
    relative_bounding_box: RelativeBoundingBox


class FaceProtocol(Protocol):
    label_id: float
    score: list[float]
    location_data: LocationData
