import cv2
import numpy as np
import os

def calculateSingleCircleMean(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.6, 1000,
                                        param1=50, param2=30, minRadius=25)
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            # print(pt)
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 0, 255), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 255, 255), 3)
        summa = 0
        ix = 0
        for i in range(a - 2, a + 2):
            for j in range(b - 2, b + 2):
                if img[i, j][0] != 0 and img[i, j][1] != 0 and img[i, j][2] != 0\
                        and img[i, j][0] != 255 and img[i, j][1] != 255 and img[i, j][2] != 255:
                    ix += 1
                    summa += img[i, j].astype(np.float32)
        cv2.waitKey(0)


    return summa/ix


def saveCircleMeans():
    folderNames = [name for name in os.listdir('src/images/img_src/')]
    for folderName in folderNames:
        array = []
        for i in range(1, len(os.listdir(f'src/images/img_src/{folderName}/')) + 1):
            array.append(calculateSingleCircleMean(f'src/images/img_src/{folderName}/{i}.jpg'))
        np.save(f'src/files/{folderName}', array)
# saveCircleMeans()


def calculateMultipleCircleMean(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.2, 65,
                                        param1=50, param2=30, minRadius=25, maxRadius=60)
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(img, (a, b), r, (0, 0, 255), 2)
            cv2.circle(img, (a, b), 1, (0, 255, 255), 3)
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
        cv2.imshow('Detected Circle', img)
        cv2.waitKey(0)
# calculateMultipleCircleMean('src/images/others/Regisajat/sajat2.jpg')
# calculateMultipleCircleMean('src/images/others/Regisajat/sajat1.jpg')
# calculateMultipleCircleMean('src/images/others/Regisajat/sajat4.jpg')



def declarationORB(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return des

def descriptorsReference():
    folderNames = [name for name in os.listdir('src/images/img_src/')]
    for folderName in folderNames:
        for i in range(1, len(os.listdir(f'src/images/img_src/{folderName}/')) + 1):
            np.save(f'src/descriptors/{folderName}/{i}', declarationORB(f'src/images/img_src/{folderName}/{i}.jpg'))
# Descriptors_Reference()

def detectionORB(path):
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.2, 70,
                                        param1=50, param2=30, minRadius=25, maxRadius=60)
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        countcoinvalue = 0
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            folderNames = [name for name in os.listdir('src/descriptors/')]
            coin = img[b-r:b+r, a-r:a+r]
            orb = cv2.ORB_create()
            kp = orb.detect(coin, None)
            kp, des = orb.compute(coin, kp)
            maximum = 0
            index = "Not matching"
            ismatching = False
            for folder in folderNames:
                summon = 0
                for i in range(1, len(os.listdir(f'src/descriptors/{folder}/')) + 1):
                    curr_des = np.load(f'src/descriptors/{folder}/{i}.npy')
                    # create BFMatcher object
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    # Match descriptors.
                    matches = bf.match(des, curr_des)
                    good = []
                    for n in matches:
                        good.append([n])
                    summon += len(good)
                if maximum < summon:
                    maximum = summon
                    index = folder
                    count = summon
                    ismatching = True
            coinvalue = 0
            if index == 'OT':
                coinvalue = 5
            elif index == 'TIZ':
                coinvalue = 10
            elif index == 'HUSZ':
                coinvalue = 20
            elif index == 'OTVEN':
                coinvalue = 50
            elif index == 'SZAZ':
                coinvalue = 100
            elif index == 'KETSZAZ':
                coinvalue = 200
            countcoinvalue += coinvalue
            img = cv2.putText(img, f'{index}', (a, b), cv2.FONT_HERSHEY_PLAIN,
                              1, (255, 0, 0), 2, cv2.LINE_AA)
            if ismatching:
                img = cv2.putText(img, f'{count} matches', (a, b+20), cv2.FONT_HERSHEY_PLAIN,
                                  1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(img, (a, b), r, (0, 0, 255), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 255, 255), 3)
        img = cv2.putText(img, f'The value of the image is {countcoinvalue} Ft', (10, 20), cv2.FONT_HERSHEY_PLAIN,
                          1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Detected Circle', img)
        cv2.waitKey(0)

detectionORB('src/images/own/SAJAT/2.jpg')
detectionORB('src/images/own/SAJAT/5.jpg')
