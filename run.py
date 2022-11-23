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
    img = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.2, 65,
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
            """
            slc = img[a-2:a+2, b-2:b+2]
            cv2.imshow('slc', slc)
            cv2.waitKey(0)
            print(img[a-2: a+2, b-2: b+2][0][0])
            for i in slc:
                print(i)
                if i[0] != 0 and i[1] != 0 and i[2] != 0 \
                        and i[0] != 255 and i[1] != 255 and i[2] != 255:
                    ix += 1
                    summa += i.astype(np.float32)
            """
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
                    # bigger = i + threshold
                    # less = i - threshold
                    # if (avg[0] > less[0] and avg[1] > less[1] and avg[2] > less[2]) and \
                            # (avg[0] < bigger[0] and avg[1] < bigger[1] and avg[2] < bigger[2]):
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
# Circle_Detection('src/images/own/SAJAT/2.jpg')
Circle_Detection('src/images/own/SAJAT/2.JPG')
# Circle_Detection('src/images/others/Regisajat/sajat2.jpg')
# Circle_Detection('src/images/others/Regisajat/sajat1.jpg')
# Circle_Detection('src/images/others/Regisajat/sajat4.jpg')
# Circle_Detection('src/images/others/Regisajat/sajat5.jpg')
# Circle_Detection('src/images/others/Regisajat/sajat3.jpg') # nem fut le

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

def ORB_Detection(Images_Path):
    img = cv2.imread(Images_Path, 1)
    img = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur using 3 * 3 kernel.
    gray_blurred = cv2.blur(gray, (3, 3))
    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.2, 70,
                                        param1=50, param2=30, minRadius=25, maxRadius=60)
    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            folderNames = [name for name in os.listdir('src/descriptors/')]
            coin = img[b-r:b+r, a-r:a+r]
            orb = cv2.ORB_create()
            kp = orb.detect(coin, None)
            kp, des = orb.compute(coin, kp)
            j = 0
            coins = ['husz', 'ketszaz', 'ot', 'otven', 'szaz', 'tiz']

            maximum = 0
            index = ""
            for folder in folderNames:
                # print(folder),
                summon = 0
                for i in range(1, len(os.listdir(f'src/descriptors/{folder}/')) + 1):
                    curr_des = np.load(f'src/descriptors/{folder}/{i}.npy')
                    # create BFMatcher object
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    # Match descriptors.
                    matches = bf.match(des, curr_des)
                    good = []
                    lowe_ratio = 0.89
                    for n in matches:
                        good.append(n)
                    # Sort them in the order of their distance.
                    # matches = sorted(matches, key=lambda x: x.distance)
                    summon += len(good)
                # print(summon)
                if maximum < summon:
                    maximum = summon
                    index = folder

            img = cv2.putText(img, f'{index}', (a, b), cv2.FONT_HERSHEY_PLAIN,
                              1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(img, (a, b), r, (0, 0, 255), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 255, 255), 3)

        cv2.imshow('Detected Circle', img)
        cv2.waitKey(0)

# ORB_Detection('src/images/own/SAJAT/2.jpg')
# ORB_Detection('src/images/own/SAJAT/5.jpg')
"""
ORB_Detection('src/images/own/SAJAT/1.jpg')
ORB_Detection('src/images/own/SAJAT/2.jpg') # 2-esre nem rossz
ORB_Detection('src/images/own/SAJAT/3.jpg')
ORB_Detection('src/images/own/SAJAT/4.jpg')
ORB_Detection('src/images/own/SAJAT/5.jpg')
ORB_Detection('src/images/own/SAJAT/6.jpg')
ORB_Detection('src/images/own/SAJAT/7.jpg')
"""
"""
ORB_Detection('src/images/others/Regisajat/sajat1.jpg')
ORB_Detection('src/images/others/Regisajat/sajat2.jpg')
ORB_Detection('src/images/others/Regisajat/sajat3.jpg')
ORB_Detection('src/images/others/Regisajat/sajat4.jpg')
ORB_Detection('src/images/others/Regisajat/sajat5.jpg')
"""