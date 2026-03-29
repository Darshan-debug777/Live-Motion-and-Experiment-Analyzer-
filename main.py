import cv2
import numpy as np
import time
import threading
import speech_recognition as sr
import pyttsx3

# ---------------------------
# Jarvis voice setup
# ---------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------------------
# Speech recognition
# ---------------------------
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_command():
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening... speak now!")
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
    try:
        command = recognizer.recognize_google(audio).lower()
        print(f"You said: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError:
        print("Network error")
        return ""


# ---------------------------
# Color tracking parameters
# ---------------------------
COLOR_RANGES = {
    'red': ((0, 120, 70), (10, 255, 255)),
    'green': ((36, 25, 25), (86, 255, 255)),
    'blue': ((94, 80, 2), (126, 255, 255)),
    'yellow': ((20, 100, 100), (30, 255, 255))
}

PIXEL_TO_METER = 1/200  # 1 meter = 200 pixels (adjust based on setup)

# ---------------------------
# Camera tracking
# ---------------------------
class ColorTracker:
    def __init__(self, color):
        self.color = color
        self.lower, self.upper = COLOR_RANGES[color]
        self.positions = []
        self.times = []

    def track(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            speak("Camera not found.")
            return

        prev_time = time.time()
        print(f"Tracking {self.color} object. Press 'q' to quit.")
        speak(f"Tracking {self.color} object.")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower, self.upper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            displacement_m = 0
            velocity_m_s = 0
            info_text = "No object detected"

            for cnt in contours:
                if cv2.contourArea(cnt) < 500:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cx, cy = x + w//2, y + h//2

                cur_time = time.time()
                if self.positions:
                    prev_cx, prev_cy = self.positions[-1]
                    dt = cur_time - self.times[-1]
                    if dt > 0:
                        displacement_px = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                        displacement_m = displacement_px * PIXEL_TO_METER
                        velocity_m_s = displacement_m / dt

                self.positions.append((cx, cy))
                self.times.append(cur_time)
                info_text = f"{self.color} | Disp: {displacement_m:.3f} m | Speed: {velocity_m_s:.3f} m/s"
                break  # track first detected object

            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Jarvis Color Tracker", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Summary
        if self.positions:
            total_disp_px = np.sqrt((self.positions[-1][0]-self.positions[0][0])**2 + (self.positions[-1][1]-self.positions[0][1])**2)
            total_disp_m = total_disp_px * PIXEL_TO_METER
            speeds = []
            for i in range(1, len(self.positions)):
                dx = self.positions[i][0]-self.positions[i-1][0]
                dy = self.positions[i][1]-self.positions[i-1][1]
                dt = self.times[i]-self.times[i-1]
                if dt > 0:
                    speeds.append(np.sqrt(dx**2+dy**2)*PIXEL_TO_METER/dt)
            max_speed = max(speeds) if speeds else 0
            avg_speed = sum(speeds)/len(speeds) if speeds else 0
            summary = f"Total displacement: {total_disp_m:.3f} m, Max speed: {max_speed:.3f} m/s, Avg speed: {avg_speed:.3f} m/s"
            print("\n"+summary)
            speak(summary)

        cap.release()
        cv2.destroyAllWindows()

# ---------------------------
# Main loop
# ---------------------------
while True:
    speak("Listening for 'jarvis'")
    command = listen_command()
    if "jarvis" in command:
        speak("Yes, I am here. What do you want me to do?")
        command = listen_command()
        if "track" in command:
            for color in COLOR_RANGES.keys():
                if color in command:
                    tracker = ColorTracker(color)
                    tracker.track()
                    break
            else:
                speak("Color not recognized. Try red, green, blue, or yellow.")



