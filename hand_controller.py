import cv2
import time
import HandTrackingModule as htm
import mediapipe as mp
import numpy as np
import autopy 
wCam, hCam = 640, 480
FrameR = 100
smoothening = 7
plocX, plocY = 0, 0
clocx, clocY = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(maxHands=1)
wscr, hscr = autopy.screen.size()
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList ,bbox= detector.findPosition(img)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[12][1], lmList[12][2]
        #print(x1, y1, x2, y2)
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (FrameR, FrameR), (wCam - FrameR, hCam - FrameR), (255, 0, 255), 2)
        if fingers[1]==1 and fingers[2]==0:
            x3 = np.interp(x1,(FrameR,wCam-FrameR),(0,wscr))
            y3 = np.interp(y1,(FrameR,hCam-FrameR),(0,hscr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            autopy.mouse.move(wscr-clocX,clocY)
            plocX, plocY = clocX, clocY


            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
        if fingers[1]==1 and fingers[2]==1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length<40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cv2.imshow("Image", img)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break