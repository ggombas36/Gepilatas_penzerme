import cv2
import numpy as np

def Circle_Detection(Images_Path):
    img = cv2.imread(Images_Path, 1)
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
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
		    a, b, r = pt[0], pt[1], pt[2]
		    # Draw the circumference of the circle.
		    cv2.circle(img, (a, b), r, (0, 0, 255), 2)
		    # Draw a small circle (of radius 1) to show the center.
		    cv2.circle(img, (a, b), 1, (0, 255, 255), 3)
	    cv2.imshow("Detected Circle", img)
	    cv2.waitKey(0)

# 5 forintosok:
Circle_Detection('Images/OT/ot1.jpg')
Circle_Detection('Images/OT/ot2.jpg')
Circle_Detection('Images/OT/ot3.jpg')
Circle_Detection('Images/OT/ot4.jpg')
Circle_Detection('Images/OT/ot5.jpg')
Circle_Detection('Images/OT/ot6.jpg')
Circle_Detection('Images/OT/ot7.jpg')
Circle_Detection('Images/OT/ot8.jpg')

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
Circle_Detection1('Images/SAJAT/sajat1.jpg')
Circle_Detection1('Images/SAJAT/sajat2.jpg')
Circle_Detection1('Images/SAJAT/sajat3.jpg')
Circle_Detection1('Images/SAJAT/sajat4.jpg')
Circle_Detection1('Images/SAJAT/sajat5.jpg')

