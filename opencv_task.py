import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap = cv.VideoCapture("/home/khalaq04/Downloads/tennisVid.mp4")
od = cv.createBackgroundSubtractorMOG2(history=10, varThreshold=100)

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    # print(h, w)

    roi = frame[170:850, 240:1425]
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    lower = np.array([40,40,0])
    upper = np.array([100,140,195])
    mask = cv.inRange(hsv, lower, upper)
    mask1 = od.apply(mask)
    #res = cv.bitwise_and(roi, roi, mask=mask)

    contours, hierarchy = cv.findContours(mask1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(roi, contours, -1, (0,0,255), 2)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 20 and area < 200:
            cv.drawContours(roi, [cnt], -1, (0, 0, 255), 2)


    #cv.imshow("roi", roi)
    #cv.imshow("Result", res)
    #cv.imshow("Mask", mask)
    cv.imshow("Frame", frame)

    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
