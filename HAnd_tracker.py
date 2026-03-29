import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    x1, y1 = 100, 100
    x2, y2 = 400, 400
    roi = frame[y1:y2, x1:x2]

    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold (THIS FIXES THE BLACK MASK PROBLEM)
    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Morphology clean-up
    kernel = np.ones((3,3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [c], -1, (0,255,0), 2)

        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(roi, (x,y), (x+w,y+h), (255,0,0), 2)

        cx = x + w//2
        cy = y + h//2
        cv2.circle(roi, (cx, cy), 8, (0,0,255), -1)

    cv2.imshow("Hand Tracker", frame)
    cv2.imshow("Mask", th)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


