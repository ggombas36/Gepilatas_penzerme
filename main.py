import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


img = cv2.imread('src/images/own/SAJAT/1.JPG', 1)
cv2.imshow('Detected Circle', img)
cv2.waitKey(0)
