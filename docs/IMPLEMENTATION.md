# SignSense – Implementation Details

This document explains, section by section, how the project notebook [`AIProjectSe4.ipynb`](../AIProjectSe4.ipynb) works. Section numbers match the markdown headings inside the notebook.

---

## Overview

```
Webcam ──► MediaPipe Holistic ──► 258 keypoints / frame ──► 30-frame sequence ──► LSTM ──► sign + confidence ──► on-screen text
```

| Stage | Notebook section | Key code |
|---|---|---|
| Environment | 1 | `%pip install …`, imports |
| Landmark detection | 2 | `mediapipe_detection()`, `draw_landmarks()` |
| Feature extraction | 3 | `extract_keypoints()` |
| Dataset configuration | 4 | `DATA_PATH`, `actions`, `no_sequences`, `sequence_length` |
| Data collection | 5 | webcam loop + `np.save()` |
| Pre-processing | 6 | `label_map`, `to_categorical`, `train_test_split` |
| Model + training | 7 | `Sequential` LSTM model, `model.fit()` |
| Label look-up | 8 | `actions[np.argmax(...)]` |
| Saving | 9 | `model.save('action.keras')` |
| Evaluation | 10 | `multilabel_confusion_matrix`, `accuracy_score` |
| Real-time inference | 11 | sliding window, threshold, sentence display |

---

## 1. Import and Install Dependencies

```python
%pip install tensorflow opencv-python mediapipe scikit-learn matplotlib
import cv2, numpy as np, os, time, mediapipe as mp
from matplotlib import pyplot as plt
```

| Library | Version used | Role |
|---|---|---|
| tensorflow | 2.15.0 | Builds and trains the LSTM network (Keras API) |
| opencv-python | 4.11.0.86 | Webcam capture, colour conversion, drawing text/shapes, display window |
| mediapipe | 0.10.21 | Holistic landmark model (pose, face, hands) |
| scikit-learn | 1.6.1 | `train_test_split`, evaluation metrics |
| matplotlib | 3.10.1 | Plotting library (imported) |
| numpy | 1.26.4 | Arrays and the `.npy` dataset format |

Runtime: Python 3.11.8 on Windows.

---

## 2. Keypoints using MP Holistic

```python
mp_holistic = mp.solutions.holistic        # Holistic model
mp_drawing  = mp.solutions.drawing_utils    # drawing utilities
mp_face_mesh = mp.solutions.face_mesh       # face connections
```

### `mediapipe_detection(image, model)`
1. Converts the OpenCV frame from **BGR → RGB** (OpenCV stores colour as BGR, MediaPipe expects RGB).
2. Sets `image.flags.writeable = False` before `model.process(image)` – passing a read-only image lets MediaPipe avoid copying it.
3. Sets the image back to writeable and converts **RGB → BGR** so OpenCV can draw on and display it.
4. Returns the image and the `results` object, which holds `pose_landmarks`, `face_landmarks`, `left_hand_landmarks` and `right_hand_landmarks`.

### `draw_landmarks(image, results)`
Draws landmarks and their connections on the frame with `mp_drawing.draw_landmarks`, each with its own colours (`DrawingSpec`):
- pose → `POSE_CONNECTIONS`
- left hand → `HAND_CONNECTIONS`
- right hand → `HAND_CONNECTIONS`

The face-mesh drawing line is commented out, so the face is not drawn.

### Webcam test loop
`cv2.VideoCapture(0)` opens the default camera. Inside `with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:` every frame is read, processed, drawn and shown in a window called **"OpenCV Feed"**. `cv2.waitKey(10) & 0xFF == ord('q')` exits the loop when **q** is pressed; then the camera is released and windows are closed.

- `min_detection_confidence = 0.5` – minimum confidence for MediaPipe to accept a new detection.
- `min_tracking_confidence = 0.5` – minimum confidence to keep tracking landmarks from the previous frame instead of re-detecting.

---

## 3. Extract Keypoint Values

The first cell extracts each body part from the last `results` of the webcam test, showing the size of every block:

| Block | Size | Formula |
|---|---|---|
| pose | 132 | 33 landmarks × (x, y, z, visibility) |
| face | 1,404 | 468 landmarks × (x, y, z) |
| left hand | 63 | 21 landmarks × (x, y, z) |
| right hand | 63 | 21 landmarks × (x, y, z) |

The function used everywhere else is:

```python
def extract_keypoints(results):
    pose = ... if results.pose_landmarks else np.zeros(33*4)
    # face = ...  (commented out)
    lh = ... if results.left_hand_landmarks else np.zeros(21*3)
    rh = ... if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])      # 132 + 63 + 63 = 258
```

- Each landmark list is turned into a 2-D array (landmarks × coordinates) and `.flatten()`ed into 1-D.
- **Zero padding:** when a part is not detected (e.g. a hand out of view) a zero vector of the right size is used, so every frame always produces exactly 258 numbers – the LSTM needs a fixed input size.
- **Face excluded:** the face line is commented out, so the face's 1,404 values are not part of the feature vector. The model input is therefore `(30, 258)`.
- x and y are normalised to the image width/height (0–1) by MediaPipe; z is depth relative to a reference point; `visibility` is the likelihood that a pose landmark is visible.

---

## 4. Setup Folders for Collection

```python
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30       # videos per sign
sequence_length = 30    # frames per video
start_folder = 30
```

The second cell creates `MP_Data/<action>/<sequence>` for every action and sequence number 0–29. `os.makedirs` is wrapped in `try/except: pass`, so folders that already exist are left alone. (`start_folder` is defined but not used by the later cells.)

---

## 5. Collect Keypoint Values for Training and Testing

For each action:
1. Prints a message and waits **5 seconds** (`time.sleep(5)`) so the signer can get ready.
2. Records `no_sequences` (30) sequences of `sequence_length` (30) frames.
3. On **frame 0** of every sequence the window shows **"STARTING COLLECTION"** and the code pauses **2 seconds** (`cv2.waitKey(2000)`) – a short break between repetitions.
4. Every frame shows *"Collecting frames for &lt;action&gt; Video Number &lt;n&gt;"*.
5. `extract_keypoints(results)` is saved with `np.save` to `MP_Data/<action>/<sequence>/<frame_num>.npy`.

Result: 3 × 30 × 30 = **2,700 files**, each holding one 258-value `float64` vector.

---

## 6. Preprocess Data and Create Labels and Features

```python
label_map = {label:num for num, label in enumerate(actions)}
# {'hello': 0, 'thanks': 1, 'iloveyou': 2}
```

For every action and every sequence folder, the 30 frame files are loaded in order into a `window` list; each window is appended to `sequences` and its label to `labels`.

```python
X = np.array(sequences)                    # (90, 30, 258)
y = to_categorical(labels).astype(int)     # (90, 3)  one-hot, e.g. thanks -> [0, 1, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
```

- **One-hot encoding** is required by the `categorical_crossentropy` loss and the 3-unit softmax output.
- `test_size=0.05` → **85 training** sequences and **5 test** sequences. No `random_state` is set, so the split is different each time the cell runs.

---

## 7. Build and Train LSTM Neural Network

```python
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64,  return_sequences=True,  activation='relu', input_shape=(30, 258)))
model.add(LSTM(128, return_sequences=True,  activation='relu'))
model.add(LSTM(64,  return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=700, callbacks=[tb_callback])
```

### Why an LSTM?
A sign is a **movement**, not a single pose. A Long Short-Term Memory network reads the 30 frames one after another and keeps an internal memory (cell state) controlled by input, forget and output gates, so it can learn how the keypoints change over time.

### Layer by layer

| Layer | Output | Params | Purpose |
|---|---|---|---|
| LSTM 64, `return_sequences=True` | (30, 64) | 82,688 | Learns low-level motion features at every time step and passes the full sequence on |
| LSTM 128, `return_sequences=True` | (30, 128) | 98,816 | Learns higher-level temporal patterns |
| LSTM 64, `return_sequences=False` | (64,) | 49,408 | Summarises the whole sequence into one vector (last time step) |
| Dense 64, ReLU | (64,) | 4,160 | Combines the temporal features |
| Dense 32, ReLU | (32,) | 2,080 | Further compression |
| Dense 3, Softmax | (3,) | 99 | One probability per sign; probabilities sum to 1 |
| **Total** | | **237,251** | |

Parameter count of an LSTM layer = 4 × (units × (inputs + units) + units). For the first layer: 4 × (64 × (258 + 64) + 64) = 82,688.

### Training configuration

| Setting | Value |
|---|---|
| Optimiser | Adam (Keras default learning rate) |
| Loss | Categorical cross-entropy |
| Metric | Categorical accuracy |
| Epochs | 700 |
| Batch size | Keras default (32) → 3 steps per epoch for 85 samples |
| Callback | TensorBoard → `Logs/` |

### What happened during training (from the notebook output)
- Epoch 1: loss 1.0856, accuracy 49.4 %.
- Accuracy first reached 100 % at **epoch 9**.
- The loss was unstable between epochs 13 and 62 (largest spike 15.67 at epoch 37).
- From **epoch 63** onwards accuracy stayed at 100 % and the loss kept falling smoothly to **8.13 × 10⁻⁸** at epoch 700.

---

## 8. Make Predictions

```python
actions[np.argmax(res[1])]       # -> 'hello'
actions[np.argmax(y_test[1])]    # -> 'thanks'
```

These cells show how a class index is turned back into a sign name with `np.argmax` and the `actions` array – e.g. the one-hot label `y_test[1]` maps to `'thanks'`. (The model's actual predictions on the test set are computed in Section 10.)

---

## 9. Save Weights

```python
model.save('action.keras')
```

Saves the whole model – architecture, trained weights and optimiser state – in the native Keras format. The repository includes this file as [`action.keras`](../action.keras) (it can be loaded with `tf.keras.models.load_model('action.keras')`).

---

## 10. Evaluation using Confusion Matrix and Accuracy

```python
yhat  = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()   # one-hot  -> class index
yhat  = np.argmax(yhat, axis=1).tolist()     # probabilities -> class index
multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)
```

`multilabel_confusion_matrix` gives one 2 × 2 matrix per sign in the layout `[[TN, FP], [FN, TP]]`:

| Sign | Matrix | Meaning |
|---|---|---|
| hello | `[[3, 0], [0, 2]]` | 2 hello sequences correctly found, 3 others correctly rejected |
| thanks | `[[4, 0], [0, 1]]` | 1 thanks sequence correctly found, 4 others correctly rejected |
| iloveyou | `[[3, 0], [0, 2]]` | 2 iloveyou sequences correctly found, 3 others correctly rejected |

No false positives or false negatives → `accuracy_score = 1.0` (5 / 5).

---

## 11. Test in Real Time

### State
```python
sequence = []      # last 30 keypoint vectors
sentence = []      # words shown on screen
predictions = []   # history of predicted class indexes
threshold = 0.9    # minimum confidence
```

### Loop (per webcam frame)
1. Read a frame, run `mediapipe_detection`, draw landmarks.
2. `keypoints = extract_keypoints(results)` → append to `sequence`, keep only `sequence[-30:]` (a **sliding window** – every new frame drops the oldest one).
3. Once 30 frames are available:
   - `res = model.predict(np.expand_dims(sequence, axis=0))[0]` – `expand_dims` adds the batch dimension, giving shape (1, 30, 258); `[0]` takes the 3 probabilities of that single sample.
   - The predicted sign `actions[np.argmax(res)]` is printed and its index appended to `predictions`.
4. **Sentence logic**
   - Only if the top probability `res[np.argmax(res)]` is **greater than 0.9** is the word considered.
   - It is appended only if the sentence is empty or the word is **different from the last word**, so holding a sign does not repeat it.
   - Only the last **5** words are kept.
5. **Display** – `cv2.rectangle(image, (0,0), (640,40), (245,117,16), -1)` draws a filled banner across the top, and `cv2.putText` writes `' '.join(sentence)` on it in white.
6. Press **q** to stop; the camera is released and windows closed. The last cell repeats `cap.release()` / `cv2.destroyAllWindows()` as a safety net in case the window stays open.

### Observed behaviour
The printed output of this cell shows the prediction stream following the signer through **hello → thanks → iloveyou → thanks → hello → thanks**, with each sign held over many consecutive frames, and most predictions taking about 15–40 ms each.

---

## Files produced by the notebook

| File / folder | Created by | Content |
|---|---|---|
| `MP_Data/<sign>/<seq>/<frame>.npy` | Section 5 | Keypoint dataset |
| `Logs/train/events.out.tfevents.*` | Section 7 (TensorBoard callback) | Training curves for TensorBoard |
| `action.keras` | Section 9 | Trained model |

> The `Logs/train` folder in this repository contains the TensorBoard event files from one of the team's training sessions on the same machine. The 700-epoch curves shown in the README, report and slides are plotted from the training log printed in the notebook.
