import cv2
import numpy as np


coin = cv2.imread('Images/forintok.jpg', 1)
cv2.imshow("kep", coin)
cv2.waitKey(0)
gray = cv2.imread('Images/forintok.jpg', 0)


"""
gray1 = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
cv2.imshow("kep3", gray1)
cv2.waitKey(0)
"""
