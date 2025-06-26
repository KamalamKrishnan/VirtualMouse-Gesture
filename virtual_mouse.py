import cv2
import numpy as np
import time
import autopy
import HandTrackingModule as htm
import pyautogui

#######################
wCam, hCam = 640, 480
frameR = 100  # Frame reduction for movement area
smoothening = 5
#######################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

plocX, plocY = 0, 0
clocX, clocY = 0, 0

wScr, hScr = autopy.screen.size()

# Hand detector
detector = htm.HandDetector(maxHands=1, detectionCon=0.75)
dragging = False

while True:
    # Get image from webcam
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    if len(lmList) != 0:
        fingers = detector.fingersUp()

        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip
        x_thumb, y_thumb = lmList[4][1:]
        x_pinky, y_pinky = lmList[20][1:]

        # Move Mode: Only Index Finger
        if fingers == [0, 1, 0, 0, 0]:
            # Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Smooth movement
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            autopy.mouse.move(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

        # Left Click: Index + Middle fingers up
        if fingers == [0, 1, 1, 0, 0]:
            length = detector.findDistance(8, 12, img)[0]
            if length < 40:
                autopy.mouse.click()
                time.sleep(0.2)

        # Right Click: Thumb + Pinky up
        if fingers == [1, 0, 0, 0, 1]:
            autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
            time.sleep(0.3)

        # Scroll Up: All 5 fingers up
        if fingers == [1, 1, 1, 1, 1]:
            pyautogui.scroll(20)
            time.sleep(0.2)

        # Scroll Down: Index + Middle + Ring up
        if fingers == [0, 1, 1, 1, 0]:
            pyautogui.scroll(-20)
            time.sleep(0.2)

        # Click and Drag: Index + Middle up AND close together
        if fingers == [0, 1, 1, 0, 0]:
            length = detector.findDistance(8, 12, img)[0]
            if length < 40 and not dragging:
                dragging = True
                autopy.mouse.toggle(autopy.mouse.Button.LEFT,
                                    True)  # Mouse button down
            elif length >= 40 and dragging:
                dragging = False
                autopy.mouse.toggle(autopy.mouse.Button.LEFT,
                                    False)  # Mouse button up

    else:
        if dragging:
            autopy.mouse.toggle(False)
            dragging = False

    # Display
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR,
                  hCam - frameR), (255, 0, 255), 2)
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
