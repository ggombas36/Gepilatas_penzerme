import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



def calculateMultipleCircleMean(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.2, 150,
                                        param1=50, param2=30, minRadius=25, maxRadius=60)

    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(img, (a, b), r, (0, 0, 255), 2)
            cv2.circle(img, (a, b), 1, (0, 255, 255), 3)
            """
            ix = 0
            summa = 0
            for i in range(a - 2, a + 2):
                for j in range(b - 2, b + 2):
                    if img[i, j][0] != 0 and img[i, j][1] != 0 and img[i, j][2] != 0 \
                            and img[i, j][0] != 255 and img[i, j][1] != 255 and img[i, j][2] != 255:
                        ix += 1
                        summa += img[i, j].astype(np.float32)
            avg = summa/ix
            folderNames = [name for name in os.listdir('src/files/')]
            matches = []
            for file in folderNames:
                openFile = np.load(f'src/files/{file}')
                match = 0
                threshold = 25
                for i in openFile:
                    if abs(avg[0] - i[0]) < threshold and abs(avg[1] - i[1]) < threshold \
                            and abs(avg[2] - i[2]) < threshold:
                        match += 1
                matches.append(match)
            maxMatches = 0
            index = 0
            p = 0
            for i in matches:
                if maxMatches < i:
                    maxMatches = i
                    index = p
                p += 1
            if index > 0:
                percent = index * 10
                coins = ['husz', 'ketszaz', 'ot', 'otven', 'szaz', 'tiz']
                img = cv2.putText(img, f'{coins[index]} {percent}%', (a, b), cv2.FONT_HERSHEY_PLAIN,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, 'nincs egyezes', (a, b), cv2.FONT_HERSHEY_PLAIN,
                            1, (255, 0, 0), 2, cv2.LINE_AA)
           """
        cv2.imshow('Detected Circle', img)
        cv2.waitKey(0)

calculateMultipleCircleMean('src/images/others/Regisajat/sajat2.jpg')
