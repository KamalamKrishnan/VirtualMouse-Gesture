import cv2
import numpy as np
import time
import HandTrackingModule as htm

# Setup
wCam, hCam = 640, 480
frameR = 20

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector(maxHands=1, detectionCon=0.8)

prevGesture = ""
gestureDelay = 1  # seconds
lastGestureTime = time.time()


def classify_gesture(fingers, lmList, img):
    # Use strict matching for distinct gestures first
    if fingers == [1, 1, 1, 1, 1]:
        return "Scroll Up", "Scrolling Up"

    elif fingers == [0, 1, 1, 1, 0]:
        return "Scroll Down", "Scrolling Down"

    elif fingers == [1, 0, 0, 0, 1]:
        return "Right Click", "Clicking Right"

    elif fingers == [0, 1, 1, 0, 0]:
        length = detector.findDistance(8, 12, img)[0]
        if length < 40:
            return "Click + Drag", "Dragging"
        else:
            return "Click + Drag Gesture", "Holding Fingers"

    elif fingers == [0, 1, 1, 0, 0]:
        return "Left Click Gesture", "Waiting to Click"

    elif fingers == [0, 1, 0, 0, 0]:
        return "Move Cursor", "Moving Mouse"

    return "No Gesture", "No Action"


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # <-- Flips image horizontally (mirror view)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    gesture, action = "No Gesture", "No Action"

    if len(lmList) != 0:
        fingers = detector.fingersUp()
        gesture, action = classify_gesture(fingers, lmList, img)

        # Avoid flickering display
        if gesture != prevGesture:
            now = time.time()
            if now - lastGestureTime > gestureDelay:
                prevGesture = gesture
                lastGestureTime = now
    else:
        gesture = "No Hand"
        action = "No Action"

    # Display text
    cv2.rectangle(img, (10, 30), (460, 110), (0, 0, 0), -1)
    cv2.putText(img, f'Gesture Detected: {gesture}', (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(img, f'Action: {action}', (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show boundary box for gesture area
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR,
                  hCam - frameR), (255, 0, 255), 2)

    cv2.imshow("Gesture-Only Virtual Mouse Demo", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
