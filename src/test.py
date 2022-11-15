import cv2
import numpy as np
import matplotlib.pyplot as plt



def Circle_Detection(Image_Path):
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
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 0, 255), 2)
            # Draw a small circle (of radius 1) to show the center.
            cv2.circle(img, (a, b), 1, (0, 255, 255), 3)
        # cv2.imshow("Detected Circle", detected_circles)
        cv2.imshow("Detected Circle", img)
        sum = 0
        ix = 0
        for i in range(a - 2, a + 2):
            for j in range(b - 2, b + 2):
                if img[i, j][0] != 0 and img[i, j][1] != 0 and img[i, j][2] != 0\
                        and img[i, j][0] != 255 and img[i, j][1] != 255 and img[i, j][2] != 255:
                    # print(img[i, j])
                    # print(img[i, j][0])
                    ix += 1
                    sum += img[i, j].astype(float)

        # avg = (img[a-1, b]/5 + img[a+1, b]/5 + img[a, b]/5 + img[a, b+1]/5 + img[a, b-1]/5)
        # print(sum/ix)
        cv2.waitKey(0)


    return sum/ix

# 50 forintosok:
Circle_Detection('Images/OTVEN/1.jpg')
Circle_Detection('Images/OTVEN/2.jpg')
Circle_Detection('Images/OTVEN/3.jpg')
Circle_Detection('Images/OTVEN/4.jpg')
Circle_Detection('Images/OTVEN/5.jpg')
Circle_Detection('Images/OTVEN/6.jpg')
Circle_Detection('Images/OTVEN/7.jpg')
Circle_Detection('Images/OTVEN/8.jpg')
Circle_Detection('Images/OTVEN/9.jpg')
Circle_Detection('Images/OTVEN/10.jpg')

otvenatlag = Circle_Detection('Images/OTVEN/1.jpg') +\
Circle_Detection('Images/OTVEN/2.jpg') +\
Circle_Detection('Images/OTVEN/3.jpg') +\
Circle_Detection('Images/OTVEN/4.jpg') +\
Circle_Detection('Images/OTVEN/5.jpg') +\
Circle_Detection('Images/OTVEN/6.jpg') +\
Circle_Detection('Images/OTVEN/7.jpg') +\
Circle_Detection('Images/OTVEN/8.jpg') +\
Circle_Detection('Images/OTVEN/9.jpg') +\
Circle_Detection('Images/OTVEN/10.jpg')
print(otvenatlag/10)

def SIFT_Algorithm(Image_Path):
    # ori = cv2.imread(Image_Path)
    # ori = cv2.resize(ori, (0, 0), fx=0.3, fy=0.3)
    img = cv2.imread(Image_Path)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('Original', ori)
    cv2.imshow('SIFT', img)
    cv2.waitKey(0)
    return kp, des

def Ot_Ft():
    kp1, des1 = SIFT_Algorithm('Images/OT/ot1.jpg')
    kp2, des2 = SIFT_Algorithm('Images/OT/ot2.jpg')
    kp3, des3 = SIFT_Algorithm('Images/OT/ot3.jpg')
    kp4, des4 = SIFT_Algorithm('Images/OT/ot4.jpg')
    kp5, des5 = SIFT_Algorithm('Images/OT/ot5.jpg')
    kp6, des6 = SIFT_Algorithm('Images/OT/ot6.jpg')
    kp7, des7 = SIFT_Algorithm('Images/OT/ot7.jpg')
    kp8, des8 = SIFT_Algorithm('Images/OT/ot8.jpg')

# Ot_Ft()


"""
img = cv2.imread("Images/OT/ot1.jpg", 1)
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread("Images/OT/ot2.jpg", 0)

img2 = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

sift = cv2.SIFT_create()

# bf = cv2.BFMatcher(cv2.NORM_L2, crosschek = True)

keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
kp = sift.detect(gray, None)
# matches = bf.match(descriptors_1, descriptors_2)
# matches = sorted(matches, key = lambda x: x.distance)

# img3 = cv2.drawMatches(img, keypoints_1, img2, keypoints_2, matches[300:600],flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img3 = cv2.drawKeyponits(gray, kp, img)

cv2.imshow("Detected Circle", img3)
cv2.waitKey(0)
cv2.DestroyAllWindows()"""
def Circle_Detection1(Images_Path):
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
        cv2.imshow("Detected Circle", img)
        cv2.waitKey(0)

"""
img1 = Circle_Detection('Images/OT/ot1.jpg')  # queryImage
img2 = Circle_Detection1('Images/SAJAT/sajat1.jpg')  # trainImage
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
# cv2.imshow('kep', kp2)
# cv2.waitKey(0)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)
# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3), plt.show()
"""
"""
# 5 forintosok:
Circle_Detection('Images/OT/ot1.jpg')
Circle_Detection('Images/OT/ot2.jpg')
Circle_Detection('Images/OT/ot3.jpg')
Circle_Detection('Images/OT/ot4.jpg')
Circle_Detection('Images/OT/ot5.jpg')
Circle_Detection('Images/OT/ot6.jpg')
Circle_Detection('Images/OT/ot7.jpg')
Circle_Detection('Images/OT/ot8.jpg')
"""
"""
# 10 forintosok:
Circle_Detection('Images/TIZ/tiz1.jpg')
Circle_Detection('Images/TIZ/tiz2.jpg')
Circle_Detection('Images/TIZ/tiz3.jpg')
Circle_Detection('Images/TIZ/tiz4.jpg')
Circle_Detection('Images/TIZ/tiz5.jpg')
Circle_Detection('Images/TIZ/tiz6.jpg')
Circle_Detection('Images/TIZ/tiz7.jpg')
Circle_Detection('Images/TIZ/tiz8.jpg'),

# 20 forintosok:
Circle_Detection('Images/HUSZ/husz1.jpg')
Circle_Detection('Images/HUSZ/husz2.jpg')
Circle_Detection('Images/HUSZ/husz3.jpg')
Circle_Detection('Images/HUSZ/husz4.jpg')
Circle_Detection('Images/HUSZ/husz5.jpg')
Circle_Detection('Images/HUSZ/husz6.jpg')
Circle_Detection('Images/HUSZ/husz7.jpg')
Circle_Detection('Images/HUSZ/husz8.jpg')

# 50 forintosok:
Circle_Detection('Images/OTVEN/otven1.jpg')
Circle_Detection('Images/OTVEN/otven2.jpg')
Circle_Detection('Images/OTVEN/otven3.jpg')
Circle_Detection('Images/OTVEN/otven4.jpg')
Circle_Detection('Images/OTVEN/otven5.jpg')
Circle_Detection('Images/OTVEN/otven6.jpg')
Circle_Detection('Images/OTVEN/otven7.jpg')
Circle_Detection('Images/OTVEN/otven8.jpg')

# 100 forintosok:
Circle_Detection('Images/SZAZ/szaz1.jpg')
Circle_Detection('Images/SZAZ/szaz2.jpg')
Circle_Detection('Images/SZAZ/szaz3.jpg')
Circle_Detection('Images/SZAZ/szaz4.jpg')
Circle_Detection('Images/SZAZ/szaz5.jpg')
Circle_Detection('Images/SZAZ/szaz6.jpg')
Circle_Detection('Images/SZAZ/szaz7.jpg')
Circle_Detection('Images/SZAZ/szaz8.jpg')

# 200 forintosok:
Circle_Detection('Images/KETSZAZ/ketszaz1.jpg')
Circle_Detection('Images/KETSZAZ/ketszaz2.jpg')
Circle_Detection('Images/KETSZAZ/ketszaz3.jpg')
Circle_Detection('Images/KETSZAZ/ketszaz4.jpg')
Circle_Detection('Images/KETSZAZ/ketszaz5.jpg')
Circle_Detection('Images/KETSZAZ/ketszaz6.jpg')
Circle_Detection('Images/KETSZAZ/ketszaz7.jpg')
Circle_Detection('Images/KETSZAZ/ketszaz8.jpg')
"""

"""
Circle_Detection1('Images/SAJAT/sajat1.jpg')
Circle_Detection1('Images/SAJAT/sajat2.jpg')
Circle_Detection1('Images/SAJAT/sajat3.jpg')
Circle_Detection1('Images/SAJAT/sajat4.jpg')
Circle_Detection1('Images/SAJAT/sajat5.jpg')
"""
