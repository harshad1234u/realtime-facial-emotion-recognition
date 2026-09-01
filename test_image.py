"""
Offline / Static Image Emotion Recognition Tester
==================================================
Usage:
    python test_image.py --input sample.jpg --output annotated_output.jpg
    python test_image.py --generate-synthetic
"""

import os
import sys
import argparse
import cv2
import numpy as np
from realtime_emotion import EmotionDetector, HUDVisualizer, EMOTION_STYLES

def process_image(image_path, output_path=None, backend="tflite", show_window=False):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return False

    detector = EmotionDetector(backend=backend)
    visualizer = HUDVisualizer(detector.class_names)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50)
    )

    print(f"[INFO] Detected {len(faces)} face(s) in {image_path}")

    primary_probs = None
    primary_emotion = None

    for i, (x, y, w, h) in enumerate(faces):
        padding_x = int(0.05 * w)
        padding_y = int(0.08 * h)
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(frame.shape[1], x + w + padding_x)
        y2 = min(frame.shape[0], y + h + padding_y)

        face_crop = gray[y1:y2, x1:x2]
        tensor = detector.preprocess_face(face_crop)
        probs = detector.predict(tensor)

        best_idx = np.argmax(probs)
        emotion = detector.class_names[best_idx]
        confidence = float(probs[best_idx])

        if i == 0:
            primary_probs = probs
            primary_emotion = emotion

        style = EMOTION_STYLES.get(emotion, {"color": (255, 255, 255)})
        visualizer.draw_face_hud(frame, (x, y, w, h), emotion, confidence, style["color"])
        print(f"  Face #{i+1}: Classified as '{emotion.upper()}' with {confidence*100:.2f}% confidence.")

    if primary_probs is not None:
        visualizer.draw_probability_bars(frame, primary_probs, primary_emotion)

    visualizer.draw_top_bar(frame, fps=0.0, backend=backend, show_bars=True, show_help=False)

    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"[SUCCESS] Saved annotated result to: {output_path}")

    if show_window:
        cv2.imshow("Emotion Recognition Test", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True

def create_synthetic_test():
    """Creates a synthetic test image with facial landmarks and runs inference on it to verify pipeline."""
    print("[INFO] Generating synthetic test canvas...")
    img = np.ones((720, 1280, 3), dtype=np.uint8) * 40
    # Draw simple stylized portrait for pipeline testing
    center = (640, 360)
    cv2.circle(img, center, 140, (210, 220, 230), -1)  # Face
    cv2.circle(img, (590, 320), 18, (30, 30, 30), -1)  # Left eye
    cv2.circle(img, (690, 320), 18, (30, 30, 30), -1)  # Right eye
    cv2.ellipse(img, (640, 410), (50, 30), 0, 0, 180, (20, 20, 180), -1) # Smiling mouth

    test_input_path = os.path.join(os.path.dirname(__file__), "test_synthetic_face.jpg")
    test_output_path = os.path.join(os.path.dirname(__file__), "test_synthetic_output.jpg")
    cv2.imwrite(test_input_path, img)
    print(f"[INFO] Saved synthetic input to: {test_input_path}")

    # Process directly through detector
    detector = EmotionDetector(backend="tflite")
    visualizer = HUDVisualizer(detector.class_names)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_crop = gray[220:500, 500:780]
    tensor = detector.preprocess_face(face_crop)
    probs = detector.predict(tensor)
    best_idx = np.argmax(probs)
    emotion = detector.class_names[best_idx]
    confidence = float(probs[best_idx])

    style = EMOTION_STYLES.get(emotion, {"color": (0, 255, 0)})
    visualizer.draw_face_hud(img, (500, 220, 280, 280), emotion, confidence, style["color"])
    visualizer.draw_probability_bars(img, probs, emotion)
    visualizer.draw_top_bar(img, fps=30.0, backend="tflite", show_bars=True, show_help=True)
    cv2.imwrite(test_output_path, img)
    print(f"[SUCCESS] Synthetic pipeline verified! Output written to: {test_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test emotion detection on static images")
    parser.add_argument("--input", type=str, help="Input image file path")
    parser.add_argument("--output", type=str, default="annotated_output.jpg", help="Output image file path")
    parser.add_argument("--backend", type=str, default="tflite", choices=["tflite", "keras"])
    parser.add_argument("--show", action="store_true", help="Display result in window")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate and test synthetic face")
    args = parser.parse_args()

    if args.generate_synthetic:
        create_synthetic_test()
    elif args.input:
        process_image(args.input, args.output, backend=args.backend, show_window=args.show)
    else:
        print("[INFO] No arguments specified. Running self-test with synthetic image...")
        create_synthetic_test()

if __name__ == "__main__":
    main()
