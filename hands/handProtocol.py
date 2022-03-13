from cProfile import label
from typing import Any, Protocol


class Point(Protocol):
    x: float
    y: float
    z: float


class Classification(Protocol):
    index: int
    score: float
    label: str


class Hand_landmarks(Protocol):
    landmark: list[Point]


class Multi_handedness(Protocol):
    classification: Classification


class Processed_hands(Protocol):
    multi_hand_landmarks: list[Hand_landmarks]
    multi_hand_world_landmarks: list[Hand_landmarks]
    multi_handedness: list[Multi_handedness]
