import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def Circle_Declaration(Image_Path):
    img = cv2.imread(Image_Path, 1)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.6, 1000,
                                        param1=50, param2=30, minRadius=25)
    # cv2.imshow(detected_circles)
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        # print(detected_circles)
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            # print(pt)
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 0, 255), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 255, 255), 3)
        # cv2.imshow("Detected Circle", detected_circles)

        # cv2.imshow("Detected Circle", img)
        summa = 0
        ix = 0
        for i in range(a - 2, a + 2):
            for j in range(b - 2, b + 2):
                if img[i, j][0] != 0 and img[i, j][1] != 0 and img[i, j][2] != 0\
                        and img[i, j][0] != 255 and img[i, j][1] != 255 and img[i, j][2] != 255:
                    # print(img[i, j])
                    # print(img[i, j][0])
                    ix += 1
                    summa += img[i, j].astype(np.float32)
                    # A.append(img[i, j].astype(np.float32))
        # X = np.average(np.asarray((A)), axis=1, keepdims=True)
        # np.save('files/50', A)
        # B = np.load('files/50.npy')
        # print(B)
        # print(X)
        # print(summa / ix)
        # avg = (img[a-1, b]/5 + img[a+1, b]/5 + img[a, b]/5 + img[a, b+1]/5 + img[a, b-1]/5)
        # print(sum/ix)
        cv2.waitKey(0)


    return summa/ix


def Color_Reference():
    folderNames = [name for name in os.listdir('src/images/img_src/')]
    for folderName in folderNames:
        array = []
        for i in range(1, len(os.listdir(f'src/images/img_src/{folderName}/')) + 1):
            # avg = str(Circle_Detection(f'images/img_src/{folderName}/{i}.jpg'))
            # with open('50.txt', 'a') as f:
            # f.write(avg + '\n')
            array.append(Circle_Declaration(f'src/images/img_src/{folderName}/{i}.jpg'))
        np.save(f'src/files/{folderName}', array)
# folderNames = [name for name in os.listdir('images/img_src/')]
# folderNames = ["OTVEN"]
# Color_Reference()


def Circle_Detection(Images_Path):
    img = cv2.imread(Images_Path, 1)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.2, 60,
                                        param1=50, param2=30, minRadius=25, maxRadius=60)
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 0, 255), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 255, 255), 3)
            ix = 0
            summa = 0
            for i in range(a - 2, a + 2):
                for j in range(b - 2, b + 2):
                    if img[i, j][0] != 0 and img[i, j][1] != 0 and img[i, j][2] != 0 \
                            and img[i, j][0] != 255 and img[i, j][1] != 255 and img[i, j][2] != 255:
                        # print(img[i, j])
                        # print(img[i, j][0])
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
                    bigger = i + threshold
                    less = i - threshold
                    if (avg[0] > less[0] and avg[1] > less[1] and avg[2] > less[2]) and \
                            (avg[0] < bigger[0] and avg[1] < bigger[1] and avg[2] < bigger[2]):
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
            percent = index * 10
            coins = ['husz', 'ketszaz', 'ot', 'otven', 'szaz', 'tiz']
            img = cv2.putText(img, f'{coins[index]} {percent}%', (a, b), cv2.FONT_HERSHEY_PLAIN,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Detected Circle', img)
        cv2.waitKey(0)
# Circle_Detection('src/images/own/SAJAT/sajat1.jpg')
# Circle_Detection('src/images/own/SAJAT/sajat2.jpg')
# Circle_Detection('src/images/own/SAJAT/sajat3.jpg')
# Circle_Detection('src/images/own/SAJAT/sajat4.jpg')
# Circle_Detection('src/images/own/SAJAT/sajat5.jpg')

def ORB_Declaration(Images_Path):
    img = cv2.imread(Images_Path, 1)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    return des

def Descriptors_Reference():
    folderNames = [name for name in os.listdir('src/images/img_src/')]
    for folderName in folderNames:
        for i in range(1, len(os.listdir(f'src/images/img_src/{folderName}/')) + 1):
            np.save(f'src/descriptors/{folderName}/{i}', ORB_Declaration(f'src/images/img_src/{folderName}/{i}.jpg'))
# Descriptors_Reference()

