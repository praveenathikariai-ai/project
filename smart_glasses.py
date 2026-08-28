from gettext import install
import queue
import subprocess
import threading
import time
from collections import defaultdict
 
import cv2
import matplotlib
import pip
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from ultralytics import YOLO
 

CAMERA_INDEX = 1
CONF_THRESHOLD = 0.6
SPEECH_COOLDOWN = 5.0
MAX_PENDING_PHRASES = 3
SPEECH_RATE = 1 


LEFT_EDGE = 1 / 3
RIGHT_EDGE = 2 / 3


 
# ---- distance estimation
FOCAL_LENGTH_PX = 700
 

KNOWN_WIDTHS_CM = {
    "person": 45,
    "chair": 45,
    "bottle": 7,
    "cup": 8,
    "laptop": 33,
    "backpack": 30,
    "handbag": 30,
    "suitcase": 45,
    "bicycle": 60,
    "car": 180,
    "motorcycle": 80,
    "dog": 30,
    "cat": 20,
    "tv": 90,
    "book": 15,
    "door": 80,
}
DEFAULT_WIDTH_CM = 30  
 
speech_queue = queue.Queue(maxsize=MAX_PENDING_PHRASES)
 
 
# speech thread
def say_via_powershell(text):
    """Calls Windows' built-in speech engine as a one-shot process.
 
    Each call is its own process, so there's no engine state to get
    stuck - the reason the original version went quiet after one object.
    """
    safe_text = text.replace("'", "")  # avoid breaking the PS string literal
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = {SPEECH_RATE}; "
        f"$s.Speak('{safe_text}')"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        creationflags=subprocess.CREATE_NO_WINDOW,  
        check=False,
    )
 
 
def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            speech_queue.task_done()
            break
        try:
            say_via_powershell(text)
        except Exception as e:
            print(f"Speech error: {e}")
        finally:
            speech_queue.task_done()
 
 
def speak_async(phrase):
    try:
        speech_queue.put_nowait(phrase)
    except queue.Full:
        pass 
 
 
# -- direction
def get_direction(box, frame_width):
    """Maps a detection box to a spoken direction."""
    x1, _, x2, _ = box.xyxy[0].tolist()
    center_x = (x1 + x2) / 2
 
    if center_x < frame_width * LEFT_EDGE:
        return "on your left"
    if center_x > frame_width * RIGHT_EDGE:
        return "on your right"
    return "ahead of you"
 
 
# ---- distance
def estimate_distance_cm(box, label):
    """Approximate distance using box pixel width and a known real width."""
    x1, _, x2, _ = box.xyxy[0].tolist()
    pixel_width = x2 - x1
    if pixel_width <= 0:
        return None
    real_width = KNOWN_WIDTHS_CM.get(label, DEFAULT_WIDTH_CM)
    return (real_width * FOCAL_LENGTH_PX) / pixel_width
 
 
def format_distance(distance_cm):
    if distance_cm is None:
        return "unknown distance"
    if distance_cm < 100:
        return f"{distance_cm:.0f} cm"
    return f"{distance_cm / 100:.1f} m"
 
 
def open_camera():
    for index in (CAMERA_INDEX, 0, 2):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"\U0001f4f7 Camera opened on index {index}")
            return cap
        cap.release()
    return None




 
# ---- session report
def generate_session_report(timestamps, fps_history, count_history, distances_by_class):
    """Saves HUD-style session graphs to PNG. Not accuracy metrics (no ground
    truth exists in live use) - this is performance and detection logging,
    useful evidence of real-time behaviour for your dissertation write-up.
    """
    if not timestamps:
        print("No frames captured, skipping report.")
        return
 
    stamp = time.strftime("%Y%m%d_%H%M%S")
    elapsed = [t - timestamps[0] for t in timestamps]
 
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
 
    axes[0, 0].plot(elapsed, fps_history, color="tab:blue")
    axes[0, 0].set_title("FPS over time")
    axes[0, 0].set_xlabel("seconds")
    axes[0, 0].set_ylabel("FPS")
 
    axes[0, 1].plot(elapsed, count_history, color="tab:orange")
    axes[0, 1].set_title("Objects detected per frame")
    axes[0, 1].set_xlabel("seconds")
    axes[0, 1].set_ylabel("count")
 
    if distances_by_class:
        classes = list(distances_by_class.keys())
        avg_distances = [sum(v) / len(v) for v in distances_by_class.values()]
        axes[1, 0].bar(classes, avg_distances, color="tab:green")
        axes[1, 0].set_title("Average distance by class (cm)")
        axes[1, 0].tick_params(axis="x", rotation=45)
 
        freq = [len(v) for v in distances_by_class.values()]
        axes[1, 1].bar(classes, freq, color="tab:red")
        axes[1, 1].set_title("Detection frequency by class")
        axes[1, 1].tick_params(axis="x", rotation=45)
    else:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
 
    fig.tight_layout()
    out_path = f"session_report_{stamp}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\U0001f4c8 Session report saved to {out_path} (open it in VS Code's file explorer)")
 
 
# -------- main
def run_live_detection():
    print("\U0001f504 Loading YOLOv8 nano model...")
    model = YOLO("yolov8n.pt")
 
    print("\U0001f4f7 Opening webcam...")
    cap = open_camera()
    if cap is None:
        print("❌ Could not open any webcam. Close other apps using it.")
        return
 
    tts_thread = threading.Thread(target=speech_worker, daemon=True)
    tts_thread.start()
 
    print("✅ Running. Press 'q' in the video window to quit.")
    last_spoken = {}  # one cooldown timer per (label, direction) pair
 
    # session logging for the end-of-run report
    timestamps = []
    fps_history = []
    count_history = []
    distances_by_class = defaultdict(list)
 
    try:
        while cap.isOpened():
            loop_start = time.time()
 
            success, frame = cap.read()
            if not success:
                print("⚠️  Empty frame from camera.")
                break
 
            frame_width = frame.shape[1]
 
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            result = results[0]
            annotated_frame = result.plot()
 
            now = time.time()
 
            # Build per-box info once: label, direction, distance.
            boxes_info = []
            for box in result.boxes:
                label = model.names[int(box.cls[0])]
                direction = get_direction(box, frame_width)
                distance_cm = estimate_distance_cm(box, label)
                boxes_info.append({"box": box, "label": label, "direction": direction, "distance_cm": distance_cm})
                if distance_cm is not None:
                    distances_by_class[label].append(distance_cm)
 
            # Draw a distance label under each box.
            for info in boxes_info:
                x1, _, _, y2 = map(int, info["box"].xyxy[0].tolist())
                cv2.putText(
                    annotated_frame,
                    format_distance(info["distance_cm"]),
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
 
            # Speak the closest object per (label, direction), respecting cooldown.
            closest_by_key = {}
            for info in boxes_info:
                key = (info["label"], info["direction"])
                current = closest_by_key.get(key)
                if current is None or (info["distance_cm"] or float("inf")) < (current["distance_cm"] or float("inf")):
                    closest_by_key[key] = info
 
            for key, info in closest_by_key.items():
                if now - last_spoken.get(key, 0.0) >= SPEECH_COOLDOWN:
                    last_spoken[key] = now
                    distance_text = format_distance(info["distance_cm"])
                    phrase = f"{info['label']} detected {info['direction']}, {distance_text} away"
                    speak_async(phrase)
                    print(f"\U0001f50a {phrase}")
 
            # ---- HUD overlay: FPS, inference time, object count ----
            loop_time = time.time() - loop_start
            fps = 1.0 / loop_time if loop_time > 0 else 0.0
            inference_ms = result.speed.get("inference", 0.0)
            detection_count = len(result.boxes)
 
            hud_lines = [
                f"FPS: {fps:.1f}",
                f"Inference: {inference_ms:.1f} ms",
                f"Objects: {detection_count}",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(
                    annotated_frame, line, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                )
 
            timestamps.append(now)
            fps_history.append(fps)
            count_history.append(detection_count)
 
            cv2.imshow("Smart Glasses Assistant", annotated_frame)
 
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\U0001f6d1 Shutting down...")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech_queue.put(None)
        tts_thread.join(timeout=5)
        generate_session_report(timestamps, fps_history, count_history, distances_by_class)
 
 
if __name__ == "__main__":
    run_live_detection()