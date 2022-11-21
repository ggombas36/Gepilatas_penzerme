import cv2
import argparse
import numpy as np

"""
coin = cv2.imread('Images/forintok.jpg', 1)
cv2.imshow("kep", coin)
cv2.waitKey(0)
gray = cv2.imread('Images/forintok.jpg', 0)
"""

"""
gray1 = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
cv2.imshow("kep3", gray1)
cv2.waitKey(0)
"""
"""
# load the image, clone it for output, and then convert it to grayscale<font></font>
image = cv2.imread('Images/forintok.jpg', 1)
# image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2) #saját kép transzformációja
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# detect circles in the image
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 142)

# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	# show the output image
	cv2.imshow("output", np.hstack([image, output]))
cv2.waitKey(0)
"""

# Read image.

img = cv2.imread('Images/1.jpg', 1)

img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
# Convert to grayscale.

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.

gray_blurred = cv2.blur(gray, (3, 3))

# Apply Hough transform on the blurred image.

detected_circles = cv2.HoughCircles(gray_blurred,

									cv2.HOUGH_GRADIENT, 1, 20, param1=50,
									param2=30,
									minRadius=1, maxRadius=40)

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

