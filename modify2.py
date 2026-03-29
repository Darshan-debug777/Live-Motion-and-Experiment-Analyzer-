"""
FULL Motion & Experiment Analyzer
Dependencies (install for Python 3.11):
  pip install mediapipe opencv-python numpy matplotlib

Features:
- Mediapipe hand tracking for robust gestures
- Kalman filter smoothing for centroid (cv2.KalmanFilter)
- Velocity & acceleration computation (pixel->meter scale configurable)
- Trajectory prediction (linear extrapolation + Kalman forecast)
- Path heatmap (saved as PNG)
- Automatic experiment detection (pendulum/projectile/linear)
- Save CSV of time,x,y,v,a and snapshot PNG
- Final plots after exit (no live plotting to avoid lag)
- Gestures:
    * Open palm (all fingers up) -> Toggle Start/Pause
    * Fist (no fingers up) -> Clear trajectory
    * Swipe right (fast hand move) -> Cycle mode / action
    * Pinch (thumb+index close) -> Save snapshot/data
    * Q key -> Quit instantly
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import csv
import os
import matplotlib.pyplot as plt
from collections import deque

# ---------------------------
# CONFIG / TUNABLES
# ---------------------------
FRAME_W, FRAME_H = 1280, 720
FPS_TARGET = 30

# Pixel-to-meter scale (calibrate by measuring known object in pixels)
PIXEL_TO_METER = 1 / 200.0  # 200 px == 1 meter (adjust for your camera/setup)

# Kalman filter noise settings
KALMAN_STATE_DIM = 4  # x, y, vx, vy
KALMAN_MEAS_DIM = 2   # x, y

# Trajectory buffers
TRAJ_MAXLEN = 2000

# Gesture detection params
SWIPE_WINDOW_SEC = 0.5
SWIPE_THRESHOLD_PX = 150
PINCH_DIST_THRESHOLD = 0.04  # normalized distance (mediapipe landmarks normalized)
GESTURE_COOLDOWN = 0.9

# Output folders
OUT_DIR = "analysis_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def now(): return time.time()

def normalized_distance(lm1, lm2, img_w, img_h):
    """Return pixel distance between two normalized landmarks."""
    x1, y1 = int(lm1.x * img_w), int(lm1.y * img_h)
    x2, y2 = int(lm2.x * img_w), int(lm2.y * img_h)
    return math.hypot(x2 - x1, y2 - y1)

def fingers_up(hand_landmarks):
    """
    Return list of booleans for thumb,index,middle,ring,pinky indicating 'up' or extended.
    Uses simple y-position checks relative to landmarks — works for most frontal poses.
    """
    tips = [4, 8, 12, 16, 20]  # thumb tip, index tip, middle tip, ring tip, pinky tip
    pip = [2, 6, 10, 14, 18]   # corresponding lower knuckles
    up = []
    for t, p in zip(tips, pip):
        up.append(hand_landmarks.landmark[t].y < hand_landmarks.landmark[p].y)
    return up

def setup_kalman():
    """Initialize an OpenCV Kalman filter for 2D position + velocities."""
    kf = cv2.KalmanFilter(KALMAN_STATE_DIM, KALMAN_MEAS_DIM)
    # State: [x, y, vx, vy]
    # Measurement: [x, y]
    kf.transitionMatrix = np.array([[1, 0, 1/ FPS_TARGET, 0],
                                    [0, 1, 0, 1/ FPS_TARGET],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], dtype=np.float32)
    # tuning covariances
    kf.processNoiseCov = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 1e-3
    kf.measurementNoiseCov = np.eye(KALMAN_MEAS_DIM, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.1
    return kf

def predict_future(kf, steps=5):
    """
    Predict future positions by rolling the Kalman filter's state forward
    without updating (copy to avoid changing original).
    """
    kf_copy = cv2.KalmanFilter()  # we'll create a copy manually
    # cheap way: extract state and use linear extrapolation
    state = kf.statePost.flatten() if kf.statePost is not None else None
    if state is None:
        return []
    predictions = []
    dt = 1.0 / FPS_TARGET
    x, y, vx, vy = state[0], state[1], state[2], state[3]
    for i in range(1, steps + 1):
        t = dt * i
        px = x + vx * t
        py = y + vy * t
        predictions.append((px, py))
    return predictions

def save_csv(filename, rows, header):
    with open(filename, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"CSV saved: {filename}")

def heatmap_from_points(points, size=(FRAME_H, FRAME_W)):
    """
    Create a heatmap image from list of (x,y) pixel coords.
    Returns heatmap as BGR image.
    """
    heat = np.zeros((size[0], size[1]), dtype=np.float32)
    for x, y in points:
        if 0 <= int(y) < size[0] and 0 <= int(x) < size[1]:
            heat[int(y), int(x)] += 1.0
    # gaussian blur to spread
    heat = cv2.GaussianBlur(heat, (0,0), sigmaX=15, sigmaY=15)
    # normalize
    heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # apply colormap
    heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
    return heat_color

def detect_experiment_type(points):
    """
    Very simple heuristic:
      - If path roughly circular => 'rotational' or 'circular'
      - If path forms a single arc and has peak => 'projectile'
      - If path oscillatory (left-right periodic) => 'pendulum'
      - Else 'linear/unknown'
    Returns a string (one of above).
    """
    if len(points) < 20:
        return "unknown"

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    # circle test: check variance of distance from centroid
    cx, cy = np.mean(xs), np.mean(ys)
    d = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    d_std_rel = np.std(d) / (np.mean(d) + 1e-6)
    if d_std_rel < 0.2 and np.mean(d) > 10:
        return "circular"

    # projectile: x increases monotonically and y has a clear peak (parabolic)
    monotonic_x = np.all(np.diff(xs) > -2) or np.all(np.diff(xs) < 2)  # roughly monotonic
    y_peak_idx = np.argmax(ys)  # note: y increases downward in image coords
    if monotonic_x and 3 < y_peak_idx < len(ys)-3:
        return "projectile"

    # pendulum: periodic x movement (detect sign changes in dx)
    dx = np.diff(xs)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(dx))) > 0)
    if zero_crossings >= 4:
        return "pendulum"

    return "linear"

# ---------------------------
# MAIN APPLICATION
# ---------------------------
def main():
    # initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    kf = setup_kalman()
    data_rows = []  # to store time, x(px), y(px), vx(m/s), vy(m/s), speed(m/s), accel(m/s^2)
    traj = []       # raw trajectory in pixel coords
    timestamps = []
    gesture_history = []
    last_gesture_time = 0

    running = False  # start paused so you can position object
    start_time = now()

    # For swipe detection using wrist x movement
    wrist_history = deque(maxlen=30)
    wrist_times = deque(maxlen=30)

    # mediapipe hands
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6) as hands:

        print("Starting — show hand gestures to control. Q to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            img_h, img_w = frame.shape[:2]

            # Convert for mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_present = False
            cen_px = None

            # If hand detected, compute palm center (average of key landmarks)
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                hand_present = True

                # calculate palm center using landmarks 0(wrist), 9(center), 5,13 knuckles
                pts = [hand.landmark[i] for i in (0, 9, 5, 13)]
                cx = int(np.mean([p.x for p in pts]) * img_w)
                cy = int(np.mean([p.y for p in pts]) * img_h)
                cen_px = (cx, cy)
                traj.append(cen_px)
                timestamps.append(now() - start_time)

                # Kalman predict & update
                meas = np.array([[np.float32(cx)], [np.float32(cy)]])
                kf.predict()
                kf.correct(meas)
                state = kf.statePost.flatten()
                kx, ky, kvx, kvy = state[0], state[1], state[2], state[3]

                # compute real-world velocities (m/s)
                vx_m_s = kvx * PIXEL_TO_METER * FPS_TARGET
                vy_m_s = kvy * PIXEL_TO_METER * FPS_TARGET
                speed = math.hypot(vx_m_s, vy_m_s)

                # compute acceleration using last velocity if available
                if len(data_rows) >= 1:
                    last_vx, last_vy, last_time = data_rows[-1][3], data_rows[-1][4], data_rows[-1][0]
                    dt = (now() - start_time) - last_time if (now() - start_time) - last_time > 1e-6 else 1e-6
                    ax = (vx_m_s - last_vx) / dt
                    ay = (vy_m_s - last_vy) / dt
                else:
                    ax = ay = 0.0

                # Store (we keep time since start, x(px), y(px), vx(m/s), vy(m/s), speed, accel_magnitude)
                data_rows.append([now() - start_time, kx, ky, vx_m_s, vy_m_s, speed, math.hypot(ax, ay)])

                # draw predicted future small ghost
                preds = predict_future(kf, steps=5)
                for p in preds:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 4, (200, 200, 255), -1)

                # draw current centroid
                cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 255), -1)

                # draw trajectory
                for i in range(1, len(traj)):
                    cv2.line(frame, traj[i-1], traj[i], (255, 255, 0), 2)

                # collect wrist landmark for swipe detection (landmark 0 = wrist)
                wrist = hand.landmark[0]
                wrist_x_px = int(wrist.x * img_w)
                wrist_y_px = int(wrist.y * img_h)
                wrist_history.append(wrist_x_px)
                wrist_times.append(now())

                # Gesture detection using finger up logic and pinch distance
                up = fingers_up(hand)
                thumb_idx_dist = normalized_distance(hand.landmark[4], hand.landmark[8], img_w, img_h) / max(img_w, img_h)

                # Raised-finger open palm if most fingers up
                is_open_palm = sum(up[1:]) >= 3  # index+middle+ring+pinkie mostly up
                is_fist = sum(up) == 0
                is_pinch = thumb_idx_dist < PINCH_DIST_THRESHOLD

                # Rising-edge gesture detection with cooldown
                current_time = now()
                if is_open_palm and (current_time - last_gesture_time) > GESTURE_COOLDOWN:
                    running = not running
                    last_gesture_time = current_time
                    print("✋ Open palm detected - toggled running ->", running)
                elif is_fist and (current_time - last_gesture_time) > GESTURE_COOLDOWN:
                    traj.clear()
                    data_rows.clear()
                    last_gesture_time = current_time
                    print("✊ Fist detected - cleared data")
                elif is_pinch and (current_time - last_gesture_time) > GESTURE_COOLDOWN:
                    # Save snapshot and CSV
                    snap_name = os.path.join(OUT_DIR, f"snapshot_{int(current_time)}.png")
                    cv2.imwrite(snap_name, frame)
                    csv_name = os.path.join(OUT_DIR, f"data_{int(current_time)}.csv")
                    save_csv(csv_name, data_rows, ["t(s)","x(px)","y(px)","vx(m/s)","vy(m/s)","speed(m/s)","accel(m/s2)"])
                    last_gesture_time = current_time
                    print("🤏 Pinch detected - saved snapshot & CSV")

                # Swipe detection: detect fast right movement
                # analyze wrist history in last SWIPE_WINDOW_SEC
                if len(wrist_history) >= 6:
                    # check recent window
                    t_now = now()
                    # find indices in wrist_times within window
                    xs = np.array(wrist_history)
                    dt = np.array([t_now - t for t in wrist_times])
                    # take last few points
                    recent_xs = xs[-8:]
                    dx = recent_xs[-1] - recent_xs[0]
                    if dx > SWIPE_THRESHOLD_PX and (now() - last_gesture_time) > GESTURE_COOLDOWN:
                        mode = (len(gesture_history) + 1) % 4
                        gesture_history.append(('swipe_right', now()))
                        last_gesture_time = now()
                        print("👉 Swipe RIGHT detected - (mode cycle)")

            # HUD overlay
            cv2.rectangle(frame, (0,0), (420,120), (10,10,10), -1)
            cv2.putText(frame, f"Running: {running}", (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
            cv2.putText(frame, f"Points: {len(traj)}", (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200),1)
            cv2.putText(frame, "Open palm: toggle | Fist: clear | Pinch: save", (12,95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180),1)

            cv2.imshow("Motion Analyzer (Full)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                # manual save
                stamp = int(now())
                cv2.imwrite(os.path.join(OUT_DIR, f"snapshot_manual_{stamp}.png"), frame)
                save_csv(os.path.join(OUT_DIR, f"data_manual_{stamp}.csv"), data_rows, ["t(s)","x(px)","y(px)","vx(m/s)","vy(m/s)","speed(m/s)","accel(m/s2)"])
                print("Manual save done")

    cap.release()
    cv2.destroyAllWindows()

    # Final analysis & outputs
    if len(traj) > 1:
        # Save CSV and snapshot
        tstamp = int(now())
        csv_file = os.path.join(OUT_DIR, f"final_data_{tstamp}.csv")
        save_csv(csv_file, data_rows, ["t(s)","x(px)","y(px)","vx(m/s)","vy(m/s)","speed(m/s)","accel(m/s2)"])

        # heatmap
        heat = heatmap_from_points(traj, size=(FRAME_H, FRAME_W))
        heat_path = os.path.join(OUT_DIR, f"heatmap_{tstamp}.png")
        cv2.imwrite(heat_path, heat)
        print("Heatmap saved:", heat_path)

        # automatic experiment detection
        exp_type = detect_experiment_type(traj)
        print("Detected experiment type:", exp_type)

        # Plot displacement & speed
        times = [r[0] for r in data_rows]
        speeds = [r[5] for r in data_rows]
        xs = [r[1] for r in data_rows]
        ys = [r[2] for r in data_rows]

        # Position plot
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(times, xs, label='x (px)')
        plt.plot(times, ys, label='y (px)')
        plt.legend(); plt.xlabel("Time (s)"); plt.title("Position (px)")

        # Speed plot
        plt.subplot(1,2,2)
        plt.plot(times, speeds, label='speed (m/s)')
        plt.legend(); plt.xlabel("Time (s)"); plt.title("Speed (m/s)")

        plt.tight_layout()
        plt.show()

    print("Done. Outputs are in:", OUT_DIR)

if __name__ == "__main__":
    main()



