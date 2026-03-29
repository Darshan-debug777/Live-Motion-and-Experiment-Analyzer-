import cv2
import numpy as np
import time                            #py -3.13 "motion and experiment analyzer.py"

import matplotlib.pyplot as plt


COLOR_RANGES = {
    'red': ((0, 120, 70), (10, 255, 255)),
    'green': ((36, 25, 25), (86, 255, 255)),
    'blue': ((94, 80, 2), (126, 255, 255)),
    'yellow': ((20, 100, 100), (30, 255, 255)),
    'black': ((0, 0, 0), (180, 255, 40))
}

PIXEL_TO_METER = 1 / 200
MIN_AREA = 500



class ColorTracker:
    def __init__(self, color):
        self.color = color
        self.lower, self.upper = COLOR_RANGES[color]
        self.positions = []
        self.times = []
        self.last_time = None
        self.fps_time = time.time()

    def track(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not found.")
            return

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        displacement_list, speed_list, time_list = [], [], []

        print(f"\n🟢 Jarvis: Tracking '{self.color}' object... Press 'q' to stop.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower, self.upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            displacement_m, velocity_m_s = 0, 0
            info_text = "No object detected"

           
            now = time.time()
            fps = 1 / (now - self.fps_time)
            self.fps_time = now

            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2

                cur_time = time.time()

               
                if self.positions:
                    prev_cx, prev_cy = self.positions[-1]
                    dt = cur_time - self.last_time if self.last_time else 0
                    if dt > 0:
                        displacement_px = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                        displacement_m = displacement_px * PIXEL_TO_METER
                        velocity_m_s = displacement_m / dt

                
                self.positions.append((cx, cy))
                self.times.append(cur_time)
                self.last_time = cur_time

                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                for i in range(1, len(self.positions[-15:])):  # Trail of last 15 points
                    cv2.line(frame, self.positions[-i - 1], self.positions[-i], (255, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

                info_text = f"{self.color} | Disp: {displacement_m:.3f} m | Speed: {velocity_m_s:.3f} m/s"
                break

           
            cv2.rectangle(frame, (0, 0), (360, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"JARVIS | Tracking: {self.color.upper()}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, info_text, (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

            cv2.imshow("🧠 Jarvis Motion Analyzer", frame)

            
            if len(self.positions) > 1:
                total_disp_px = np.sqrt((self.positions[-1][0] - self.positions[0][0]) ** 2 +
                                        (self.positions[-1][1] - self.positions[0][1]) ** 2)
                total_disp_m = total_disp_px * PIXEL_TO_METER
                displacement_list.append(total_disp_m)

                dx = self.positions[-1][0] - self.positions[-2][0]
                dy = self.positions[-1][1] - self.positions[-2][1]
                dt = self.times[-1] - self.times[-2]
                speed_m_s = np.sqrt(dx ** 2 + dy ** 2) * PIXEL_TO_METER / dt if dt > 0 else 0
                speed_list.append(speed_m_s)
                time_list.append(self.times[-1] - self.times[0])

                ax1.clear()
                ax1.plot(time_list, displacement_list, color='cyan')
                ax1.set_ylabel("Displacement (m)")
                ax1.set_title(f"{self.color.capitalize()} Displacement vs Time")

                ax2.clear()
                ax2.plot(time_list, speed_list, color='magenta')
                ax2.set_ylabel("Speed (m/s)")
                ax2.set_xlabel("Time (s)")
                ax2.set_title(f"{self.color.capitalize()} Speed vs Time")
                plt.pause(0.001)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.show()

        
        self._print_summary()

    def _print_summary(self):
        if len(self.positions) < 2:
            print("\n⚠️ No motion detected to summarize.")
            return

        total_disp_px = np.sqrt((self.positions[-1][0] - self.positions[0][0]) ** 2 +
                                (self.positions[-1][1] - self.positions[0][1]) ** 2)
        total_disp_m = total_disp_px * PIXEL_TO_METER
        speeds = []
        for i in range(1, len(self.positions)):
            dx = self.positions[i][0] - self.positions[i - 1][0]
            dy = self.positions[i][1] - self.positions[i - 1][1]
            dt = self.times[i] - self.times[i - 1]
            if dt > 0:
                speeds.append(np.sqrt(dx ** 2 + dy ** 2) * PIXEL_TO_METER / dt)

        max_speed = max(speeds) if speeds else 0
        avg_speed = np.mean(speeds) if speeds else 0

        print("\n================= ✅ JARVIS TRACKING SUMMARY =================")
        print(f"🧩 Color Tracked: {self.color}")
        print(f"📏 Total Displacement: {total_disp_m:.3f} meters")
        print(f"⚡ Max Speed: {max_speed:.3f} m/s")
        print(f"🚀 Average Speed: {avg_speed:.3f} m/s")
        print("===============================================================")



def main():
    print("🧠 LIVE MOTION ANALYZER ")
    print("Available colors: red, green, blue, yellow, black")
    print("Example: track yellow | Type 'quit' to exit.\n")

    
    default_color = "yellow"
    tracker = ColorTracker(default_color)
    tracker.track()

    
    while True:
        command = input("Enter command: ").lower().strip()

        if command == "quit":
            print("🔻 Exiting Jarvis Motion Analyzer.")
            break

        if command.startswith("track"):
            found_color = None
            for color in COLOR_RANGES:
                if color in command:
                    found_color = color
                    break

            if found_color:
                tracker = ColorTracker(found_color)
                tracker.track()
            else:
                print("⚠️ Color not recognized. Try red, green, blue, yellow, or black.")
        else:
            print("❌ Invalid command. Example: 'track blue'")


if __name__ == "__main__":
    main()




