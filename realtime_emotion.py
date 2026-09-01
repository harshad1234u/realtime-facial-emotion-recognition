"""
Real-Time Facial Emotion Recognition Desktop Application
=========================================================
Features:
- Dual inference backend: TFLite (Ultra fast CPU) & Keras (.keras)
- Haar Cascade face detection with multi-face support
- Temporal smoothing (Exponential Moving Average / Rolling Queue) to eliminate prediction jitter
- Visual HUD with emotion-specific color schemes, emojis, and confidence badges
- Live probability distribution bar chart panel
- Interactive hotkeys: Snapshot ('s'), Toggle HUD ('b'), Switch Backend ('m'), Help ('h'), Quit ('q'/'ESC')
"""

import os
import sys
import time
import json
import argparse
from collections import deque
from datetime import datetime
import cv2
import numpy as np

# Emotion styling: Color palette (BGR format for OpenCV) and Emojis
EMOTION_STYLES = {
    "angry": {"color": (36, 36, 220), "emoji": "[ANGRY]", "symbol": "😠"},       # Red
    "disgust": {"color": (34, 139, 34), "emoji": "[DISGUST]", "symbol": "🤢"},    # Forest Green
    "fear": {"color": (160, 32, 240), "emoji": "[FEAR]", "symbol": "😨"},        # Purple
    "happy": {"color": (30, 215, 96), "emoji": "[HAPPY]", "symbol": "😄"},        # Emerald Green
    "neutral": {"color": (220, 220, 0), "emoji": "[NEUTRAL]", "symbol": "😐"},    # Cyan / Light Teal
    "sad": {"color": (219, 112, 147), "emoji": "[SAD]", "symbol": "😢"},         # Muted Blue / Lavender
    "surprise": {"color": (0, 215, 255), "emoji": "[SURPRISE]", "symbol": "😲"},   # Vibrant Yellow-Gold
}

DEFAULT_METADATA_PATH = os.path.join(os.path.dirname(__file__), "model_metadata.json")
DEFAULT_TFLITE_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.tflite")
DEFAULT_KERAS_PATH = os.path.join(os.path.dirname(__file__), "emotion_model.keras")


class EmotionDetector:
    """Manages model loading and inference across TFLite and Keras backends."""

    def __init__(self, metadata_path=DEFAULT_METADATA_PATH, tflite_path=DEFAULT_TFLITE_PATH, keras_path=DEFAULT_KERAS_PATH, backend="tflite"):
        self.metadata = self._load_metadata(metadata_path)
        self.class_names = self.metadata.get("class_names", [
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        ])
        self.input_shape = tuple(self.metadata.get("input_shape", [48, 48, 1]))
        self.tflite_path = tflite_path
        self.keras_path = keras_path
        self.backend = backend.lower()

        self.tflite_interpreter = None
        self.tflite_input_idx = None
        self.tflite_output_idx = None
        self.keras_model = None

        self._init_backend()

    def _load_metadata(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _init_backend(self):
        """Initializes the active inference engine."""
        if self.backend == "tflite":
            if os.path.exists(self.tflite_path):
                import tensorflow as tf
                self.tflite_interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
                self.tflite_interpreter.allocate_tensors()
                input_details = self.tflite_interpreter.get_input_details()
                output_details = self.tflite_interpreter.get_output_details()
                self.tflite_input_idx = input_details[0]["index"]
                self.tflite_output_idx = output_details[0]["index"]
                print(f"[INFO] Initialized TFLite Engine: {self.tflite_path}")
            else:
                print(f"[WARN] TFLite file not found at {self.tflite_path}, falling back to Keras.")
                self.backend = "keras"

        if self.backend == "keras":
            if self.keras_model is None and os.path.exists(self.keras_path):
                import tensorflow as tf
                print(f"[INFO] Loading Keras model: {self.keras_path} ...")
                self.keras_model = tf.keras.models.load_model(self.keras_path)
                print("[INFO] Keras Model loaded successfully.")
            elif not os.path.exists(self.keras_path) and self.tflite_interpreter is None:
                raise FileNotFoundError(f"Neither TFLite nor Keras model could be located.")

    def switch_backend(self):
        """Toggle between TFLite and Keras inference backends on the fly."""
        new_backend = "keras" if self.backend == "tflite" else "tflite"
        try:
            self.backend = new_backend
            self._init_backend()
            print(f"[INFO] Switched backend to: {self.backend.upper()}")
            return self.backend
        except Exception as e:
            print(f"[ERROR] Failed to switch to {new_backend}: {e}")
            self.backend = "tflite" if new_backend == "keras" else "keras"
            return self.backend

    def preprocess_face(self, face_crop):
        """
        Preprocesses a cropped face image:
        1. Converts to Grayscale
        2. Resizes to 48x48
        3. Converts to float32 [0.0, 255.0] (The model has a built-in Rescaling(1/255.0) layer)
        4. Reshapes to (1, 48, 48, 1)
        """
        if face_crop is None or face_crop.size == 0:
            return None

        if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        resized = cv2.resize(gray, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_AREA)
        tensor = resized.astype("float32")[np.newaxis, ..., np.newaxis]
        return tensor

    def predict(self, face_tensor):
        """Runs model inference and returns normalized probabilities for all 7 classes."""
        if face_tensor is None:
            return np.zeros(len(self.class_names))

        if self.backend == "tflite" and self.tflite_interpreter is not None:
            self.tflite_interpreter.set_tensor(self.tflite_input_idx, face_tensor)
            self.tflite_interpreter.invoke()
            preds = self.tflite_interpreter.get_tensor(self.tflite_output_idx)[0]
        elif self.keras_model is not None:
            preds = self.keras_model(face_tensor, training=False).numpy()[0]
        else:
            preds = np.zeros(len(self.class_names))

        # Model output is already Softmax probabilities. Ensure valid numerical range:
        probs = np.clip(preds.astype(np.float32), 0.0, 1.0)
        sum_p = np.sum(probs)
        if sum_p > 0:
            probs = probs / sum_p
        return probs


class TemporalSmoother:
    """Smooths probability predictions across consecutive frames to eliminate jitter."""

    def __init__(self, window_size=5, alpha=0.65):
        self.window_size = window_size
        self.alpha = alpha  # Weight for Exponential Moving Average
        self.history = deque(maxlen=window_size)
        self.ema_probs = None

    def update(self, current_probs):
        self.history.append(current_probs)
        if self.ema_probs is None:
            self.ema_probs = np.array(current_probs, dtype=np.float32)
        else:
            self.ema_probs = self.alpha * np.array(current_probs) + (1.0 - self.alpha) * self.ema_probs
        return self.ema_probs

    def reset(self):
        self.history.clear()
        self.ema_probs = None


class HUDVisualizer:
    """Renders modern UI overlays, bounding boxes, banners, and probability distribution charts."""

    def __init__(self, class_names):
        self.class_names = class_names

    def draw_corner_rect(self, img, pt1, pt2, color, thickness=2, corner_length=20):
        """Draws a sleek high-tech bounding box with highlighted corner brackets."""
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw semi-transparent bounding box outline
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Corner length bounded by box dimensions
        cl_x = min(corner_length, (x2 - x1) // 3)
        cl_y = min(corner_length, (y2 - y1) // 3)

        t = thickness
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + cl_x, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + cl_y), color, t, cv2.LINE_AA)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - cl_x, y1), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + cl_y), color, t, cv2.LINE_AA)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + cl_x, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - cl_y), color, t, cv2.LINE_AA)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - cl_x, y2), color, t, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - cl_y), color, t, cv2.LINE_AA)

    def draw_face_hud(self, frame, bbox, emotion, confidence, color):
        """Draws the primary emotion tag and confidence percentage above the detected face."""
        x, y, w, h = bbox
        self.draw_corner_rect(frame, (x, y), (x + w, y + h), color, thickness=3, corner_length=22)

        # Badge text
        label_text = f"{emotion.upper()} {confidence * 100:.1f}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.65
        thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        badge_y1 = max(0, y - text_h - 14)
        badge_y2 = y

        # Badge background banner
        cv2.rectangle(frame, (x, badge_y1), (x + text_w + 16, badge_y2), color, cv2.FILLED)
        # Contrast text (white on dark color, dark on bright color)
        text_color = (255, 255, 255) if (color[0] * 0.11 + color[1] * 0.59 + color[2] * 0.3) < 140 else (20, 20, 20)
        cv2.putText(frame, label_text, (x + 8, badge_y2 - 6), font, font_scale, text_color, thickness, cv2.LINE_AA)

    def draw_probability_bars(self, frame, probs, active_emotion):
        """Draws a real-time horizontal bar graph of all 7 emotion probabilities."""
        h, w, _ = frame.shape
        panel_w = 270
        panel_h = 245
        panel_x = w - panel_w - 20
        panel_y = 55

        # Semi-transparent dark background card
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (18, 22, 28), cv2.FILLED)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (55, 65, 80), 1, cv2.LINE_AA)

        # Title
        cv2.putText(frame, "EMOTION PROBABILITIES", (panel_x + 15, panel_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 225, 240), 1, cv2.LINE_AA)

        max_bar_w = 115
        row_y = panel_y + 52

        for i, name in enumerate(self.class_names):
            p = float(probs[i])
            style = EMOTION_STYLES.get(name, {"color": (200, 200, 200)})
            color = style["color"]
            is_active = (name == active_emotion)

            # Indicator arrow & text
            prefix = "> " if is_active else "  "
            label_text = f"{prefix}{name.capitalize():<8}"
            label_color = (255, 255, 255) if is_active else (150, 160, 175)
            cv2.putText(frame, label_text, (panel_x + 8, row_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, label_color, 1, cv2.LINE_AA)

            # Bar track (gray background)
            track_x = panel_x + 95
            cv2.rectangle(frame, (track_x, row_y + 2), (track_x + max_bar_w, row_y + 11), (40, 48, 60), cv2.FILLED)

            # Bar fill
            fill_w = int(p * max_bar_w)
            if fill_w > 0:
                cv2.rectangle(frame, (track_x, row_y + 2), (track_x + fill_w, row_y + 11), color, cv2.FILLED)

            # Percentage text
            pct_text = f"{p * 100:>4.1f}%"
            pct_color = (255, 255, 255) if is_active else (190, 200, 210)
            cv2.putText(frame, pct_text, (track_x + max_bar_w + 8, row_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, pct_color, 1, cv2.LINE_AA)

            row_y += 24

    def draw_top_bar(self, frame, fps, backend, show_bars, show_help):
        """Draws top information banner with status, FPS counter, and hotkey shortcuts."""
        w = frame.shape[1]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 18, 24), cv2.FILLED)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, 42), (w, 42), (50, 60, 75), 1, cv2.LINE_AA)

        # Title
        cv2.putText(frame, "FER-2013 Live Emotion Vision", (15, 27),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # FPS Counter
        fps_text = f"FPS: {fps:.1f}"
        fps_color = (0, 255, 128) if fps >= 20 else ((0, 215, 255) if fps >= 10 else (0, 100, 255))
        cv2.putText(frame, fps_text, (w - 280, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)

        # Backend Badge
        backend_color = (255, 180, 0) if backend == "tflite" else (0, 140, 255)
        cv2.putText(frame, f"Engine: {backend.upper()}", (w - 170, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, backend_color, 2, cv2.LINE_AA)

        # Bottom help bar if enabled
        if show_help:
            h = frame.shape[0]
            bot_overlay = frame.copy()
            cv2.rectangle(bot_overlay, (0, h - 30), (w, h), (15, 18, 24), cv2.FILLED)
            cv2.addWeighted(bot_overlay, 0.85, frame, 0.15, 0, frame)
            help_text = "[S] Snapshot  |  [B] Toggle Probs  |  [M] Switch Model  |  [H] Toggle Help  |  [Q/ESC] Quit"
            cv2.putText(frame, help_text, (20, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 195, 210), 1, cv2.LINE_AA)


def run_app(camera_index=0, backend="tflite", width=1280, height=720, window_name="Facial Emotion Recognition"):
    """Main application loop."""
    print("=" * 65)
    print("  FER-2013 Real-Time Facial Emotion Recognition System")
    print("=" * 65)

    # Initialize Emotion Engine
    detector = EmotionDetector(backend=backend)
    smoother = TemporalSmoother(window_size=6, alpha=0.6)
    visualizer = HUDVisualizer(detector.class_names)

    # Load OpenCV Haar Cascade Face Detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade XML not found at {cascade_path}")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    print(f"[INFO] Loaded Face Detector: {cascade_path}")

    # Initialize Video Capture
    print(f"[INFO] Connecting to camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[WARN] Camera {camera_index} failed to open. Trying fallback index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print(f"[ERROR] Could not open any video capture source. Exiting.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # App state toggles
    show_prob_bars = True
    show_help = True
    snapshots_dir = os.path.join(os.path.dirname(__file__), "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    # Performance FPS tracking
    fps_history = deque(maxlen=20)
    prev_time = time.time()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    print("\n[READY] Application is running! Focus the OpenCV window to interact:")
    print("  * Press 's' to capture a snapshot")
    print("  * Press 'b' to toggle probability distribution charts")
    print("  * Press 'm' to switch between TFLite and Keras engines")
    print("  * Press 'h' to toggle hotkey legend")
    print("  * Press 'q' or 'ESC' to exit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame from camera. Retrying...")
                time.sleep(0.05)
                continue

            # Flip horizontally for natural mirror effect
            frame = cv2.flip(frame, 1)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            primary_probs = None
            primary_emotion = None

            for i, (x, y, w_box, h_box) in enumerate(faces):
                # Expand box slightly for better head crop
                padding_x = int(0.05 * w_box)
                padding_y = int(0.08 * h_box)
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(frame.shape[1], x + w_box + padding_x)
                y2 = min(frame.shape[0], y + h_box + padding_y)

                face_crop = gray_frame[y1:y2, x1:x2]
                tensor = detector.preprocess_face(face_crop)
                raw_probs = detector.predict(tensor)

                if i == 0:
                    # Apply temporal smoothing to primary face
                    smooth_probs = smoother.update(raw_probs)
                    primary_probs = smooth_probs
                    best_idx = np.argmax(smooth_probs)
                    primary_emotion = detector.class_names[best_idx]
                    confidence = float(smooth_probs[best_idx])
                else:
                    best_idx = np.argmax(raw_probs)
                    emotion = detector.class_names[best_idx]
                    confidence = float(raw_probs[best_idx])

                emotion_name = detector.class_names[best_idx]
                style = EMOTION_STYLES.get(emotion_name, {"color": (255, 255, 255)})
                visualizer.draw_face_hud(frame, (x, y, w_box, h_box), emotion_name, confidence, style["color"])

            if len(faces) == 0:
                smoother.reset()

            # Render probability HUD if faces are visible
            if show_prob_bars and primary_probs is not None:
                visualizer.draw_probability_bars(frame, primary_probs, primary_emotion)

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_history.append(1.0 / dt)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0.0

            # Render top bar
            visualizer.draw_top_bar(frame, avg_fps, detector.backend, show_prob_bars, show_help)

            # Display window
            cv2.imshow(window_name, frame)

            # Process Hotkeys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("[INFO] Quit signal received. Closing application...")
                break
            elif key == ord('s') or key == ord('S'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(snapshots_dir, f"emotion_snapshot_{timestamp}.jpg")
                cv2.imwrite(snap_path, frame)
                print(f"[SUCCESS] Snapshot saved to: {snap_path}")
            elif key == ord('b') or key == ord('B'):
                show_prob_bars = not show_prob_bars
                print(f"[INFO] Probability Bars: {'ENABLED' if show_prob_bars else 'DISABLED'}")
            elif key == ord('m') or key == ord('M'):
                new_b = detector.switch_backend()
                smoother.reset()
            elif key == ord('h') or key == ord('H'):
                show_help = not show_help

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released and windows closed. Done.")


def main():
    parser = argparse.ArgumentParser(description="Real-Time Facial Emotion Recognition Desktop App")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--backend", type=str, default="tflite", choices=["tflite", "keras"],
                        help="Inference engine backend (default: tflite)")
    parser.add_argument("--width", type=int, default=1280, help="Window display width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Window display height (default: 720)")
    args = parser.parse_args()

    run_app(camera_index=args.camera, backend=args.backend, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
