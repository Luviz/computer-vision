# Hand Gesture Recognizer

Using media pipe we can extract hand landmarks, the landmarks
relative position can be used to recognizer gestures.

## Preparing the model

The landmarks received from media pipe are relative to the screen size.
This could lead to issue, where if sufficient training data is
not provided the model may just decided that a cluster of landmark
in center is closed hand.

In order to deal with this issue, we can move the hand to top left
corner by subtracting ${[x_{min}, y_{min}]}$.

```py
landmarks = landmarks - np.min(landmarks)
```

This solution work grate, but I soon realized that the data could
be effect by distance from camera.

I dealt with this issue by scaling the landmark with:

$$
\frac{1}{[x_{max}, y_{max}]}
$$

```py
def normalize_hand(hand: np.ndarray):
    arr = hand - np.min(hand, axis=0)
    return arr / np.max(arr, axis=0)
```

This resulted in better recognition.

```mermaid

flowchart LR
    IMG(Image)
    MPI[mediaPipe.Hands.Process]
    LMK[get_landmarks]
    NHD[normalize_hand]

    IMG --> MPI --> LMK
    LMK --> NHD

```
