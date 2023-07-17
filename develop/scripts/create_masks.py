import math
import cv2 as cv
import numpy as np

saving_path = '/Masks/mask_'


mask_height = 2800
mask_width = 5320


offsets = [160, 330, 500, 660]
coeffs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]



def rotation(points, angle, radius):

    points_R = []

    for p in points:

        x = p[0]
        y = p[1]

        x_R = x*math.cos(angle) - y*math.sin(angle) + radius*math.sin(angle) + mask_width/2
        y_R = y*math.cos(angle) + x*math.sin(angle) - radius*math.cos(angle) + mask_height/2

        points_R.append([x_R,y_R])

    return points_R


def mask(coeff, offset):

    black_image = np.zeros((mask_height, mask_width, 2), dtype=np.uint8)

    temp = math.atan(mask_width / mask_height)

    angles = [ -temp, temp, math.pi - temp, math.pi + temp]

    points = []

    contour = [[x ,coeff*x*x] for x in range(-int(mask_width/2),int(mask_width/2))]

    for c in contour:
        if c[1] < mask_height:
            for j in range(int(c[1]), mask_height):
                    points.append([c[0],j])

    for angle in angles:

        points_R = rotation(points, angle, -offset)

        for p in points_R:
            if 0 < int(p[0]) < mask_width:
                if 0 < int(p[1]) < mask_height:
                    black_image[int(p[1])][int(p[0])] = 255

    kernel_closing = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

    return cv.morphologyEx(black_image, cv.MORPH_CLOSE, kernel_closing)

