import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os
from collections import deque

# ------------------------------
# Configuration
# ------------------------------
COLOR_RANGES = {
    'red': ((0, 120, 70), (10, 255, 255)),
    'blue': ((100, 150, 0), (140, 255, 255)),
    'green': ((40, 70, 70), (80, 255, 255)),
    'yellow': ((20, 100, 100), (30, 255, 255))
}
COLOR_LIST = list(COLOR_RANGES.keys())
PIXEL_TO_METER = 1 / 200
MIN_AREA = 500

# Camera / performance
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS_TARGET = 60

# Trajectory & gesture buffers
TRAJECTORY_MAXLEN = 300         # how many points we keep for drawing / gesture detection
GESTURE_WINDOW_SEC = 0.9     # window to look for swipe (seconds)
SWIPE_THRESHOLD_PX = 125       # min displacement in pixels for a swipe
SWIPE_VERTICAL_TOLERANCE = 80   # allow some vertical drift for horizontal swipes
GESTURE_COOLDOWN = 0.9          # seconds between gesture recognitions

# Snapshot folder
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ------------------------------
# Helper functions
# ------------------------------
def now():
    return time.time()

def save_snapshot(frame):
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{ts}.png")
    cv2.imwrite(filename, frame)
    print(f"📸 Snapshot saved: {filename}")

def cycle_color(current_color):
    idx = COLOR_LIST.index(current_color)
    next_idx = (idx + 1) % len(COLOR_LIST)
    return COLOR_LIST[next_idx]

# ------------------------------
# Main tracker with swipe gestures
# ------------------------------
def run_tracker(initial_color='red'):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        print("❌ Camera not found.")
        return

    track_color = initial_color
    lower, upper = COLOR_RANGES[track_color]
    trajectory = deque(maxlen=TRAJECTORY_MAXLEN)  # store (x,y)
    timestamps = deque(maxlen=TRAJECTORY_MAXLEN)  # store times
    last_gesture_time = 0

    kernel = np.ones((5,5), np.uint8)
    start_time = now()
    print("\n🧠 Jarvis Gesture Mode ACTIVE")
    print("Swipe Right -> change color | Swipe Up -> clear trajectory | Swipe Down -> save snapshot")
    print("Press 'q' to quit instantly.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # mirror for natural gestures
        display = frame.copy()

        # Smooth input
        blurred = cv2.GaussianBlur(frame, (7,7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # mask for current color
        lower, upper = COLOR_RANGES[track_color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx = cy = None 
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > MIN_AREA:
                x,y,w,h = cv2.boundingRect(c)
                cx = int(x + w/2)
                cy = int(y + h/2)
                trajectory.append((cx, cy))
                timestamps.append(now())
                # draw box & center
                cv2.rectangle(display, (x,y), (x+w, y+h), (0,255,200), 2)
                cv2.circle(display, (cx, cy), 6, (0,100,255), -1)

        # draw trajectory (smooth polyline)
        if len(trajectory) > 1:
            pts = np.array(trajectory, dtype=np.int32)
            # draw a fading trail
            for i in range(1, len(pts)):
                alpha = int(255 * (i / len(pts)))
                color = (alpha//2 + 50, alpha//3 + 100, 255 - alpha//2)
                cv2.line(display, tuple(pts[i-1]), tuple(pts[i]), color, 2)

        # HUD
        cv2.rectangle(display, (0,0), (420,120), (18,18,18), -1)
        cv2.putText(display, f"Jarvis Gesture Tracker", (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
        cv2.putText(display, f"Color: {track_color.upper()}", (12,58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,120), 2)
        cv2.putText(display, "Swipe R → color | Swipe Up → clear | Swipe Down → snapshot", (12,90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
        cv2.putText(display, "Press 'Q' to exit", (12,112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        cv2.imshow("Jarvis Motion Tracker (Gesture Mode)", display)

        # Gesture detection logic: analyze recent trajectory within window
        triggered = False
        if len(trajectory) >= 4:
            # consider points within GESTURE_WINDOW_SEC
            t_now = now()
            # collect indices in the buffer that are within the time window
            recent_pts = []
            recent_ts = []
            # iterate from newest to oldest
            for p, tstamp in zip(reversed(trajectory), reversed(timestamps)):
                if (t_now - tstamp) <= GESTURE_WINDOW_SEC:
                    recent_pts.append(p)
                    recent_ts.append(tstamp)
                else:
                    break
            # reverse to chronological
            recent_pts = recent_pts[::-1]
            recent_ts = recent_ts[::-1]

            if len(recent_pts) >= 3 and (t_now - last_gesture_time) > GESTURE_COOLDOWN:
                x_start, y_start = recent_pts[0]
                x_end, y_end = recent_pts[-1]
                dt = recent_ts[-1] - recent_ts[0] if (recent_ts[-1] - recent_ts[0])>0 else 1e-6
                dx = x_end - x_start
                dy = y_end - y_start

                # horizontal swipe (right)
                if dx > SWIPE_THRESHOLD_PX and abs(dy) < SWIPE_VERTICAL_TOLERANCE:
                    # Swipe Right detected
                    last_gesture_time = now()
                    track_color = cycle_color(track_color)
                    print(f"👉 Swipe RIGHT detected — color -> {track_color.upper()}")
                    # clear small chunk so we don't re-detect same swipe
                    trajectory.clear()
                    timestamps.clear()
                    triggered = True

                # swipe up
                elif (y_start - y_end) > SWIPE_THRESHOLD_PX and abs(dx) < SWIPE_VERTICAL_TOLERANCE:
                    last_gesture_time = now()
                    trajectory.clear()
                    timestamps.clear()
                    print("🧼 Swipe UP detected — trajectory cleared")
                    triggered = True

                # swipe down
                elif (y_end - y_start) > SWIPE_THRESHOLD_PX and abs(dx) < SWIPE_VERTICAL_TOLERANCE:
                    last_gesture_time = now()
                    save_snapshot(frame)
                    # keep trajectory intact
                    triggered = True

        # instant exit
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # After exit: show final path graph (no lag during run)
    if len(trajectory) > 1:
        x_vals = [p[0] for p in trajectory]
        y_vals = [p[1] for p in trajectory]

        plt.figure(figsize=(6,6))
        plt.plot(x_vals, y_vals, marker='o', linewidth=2)
        plt.gca().invert_yaxis()  # match image coordinate system
        plt.title("Object Motion Path (final)")
        plt.xlabel("X position (px)")
        plt.ylabel("Y position (px)")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    run_tracker(initial_color='red')
