import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
lastkey = 0
kernel = np.ones((3, 3), np.float32)/9
while(True):
    ret, frame = cap.read()
    temp = frame

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        lastkey = ord('r')
    elif key == ord('g'):
        lastkey = ord('g')
    elif key == ord('b'):
        lastkey = ord('b')
    elif key == ord('w'):
        lastkey = ord('w')
    elif key == ord('c'):
        lastkey = ord('c')
    elif key == ord('i'):
        lastkey = ord('i')

    if lastkey == ord('r'):
        frame[:, :, 0] = 0
        frame[:, :, 1] = 0
    elif lastkey == ord('b'):
        frame[:, :, 1] = 0
        frame[:, :, 2] = 0
    elif lastkey == ord('g'):
        frame[:, :, 0] = 0
        frame[:, :, 2] = 0
    elif lastkey == ord('w'):
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = np.float32(frame)
        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        dst = np.int0(dst)
        for d in dst:
            x, y = d.ravel()
            cv.circle(frame, (x, y), 3, 255, -1)
        #dst = cv.dilate(dst,None)
        #frame[dst>0.01*dst.max()]=[0,0,255]

    elif lastkey == ord('c'):
        frame = temp
    elif lastkey == ord('i'):
        frame = cv.blur(frame, (5, 5))

    cv.imshow('frame', frame)

cap.release()
cv.destroyAllWindows()
