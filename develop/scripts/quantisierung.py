import numpy as np
import cv2 as cv

img = cv.imread('/home/manai/Desktop/InStent/Github/focus_survey/focus_surver/Image_survey/Plots/Image_3/pixels_haufigkeit.png', cv.IMREAD_GRAYSCALE)

p10, p90 = np.percentile(img, (10, 90))


new_values = np.arange(0, 256, 25)

old_values = np.linspace(p10, p90, len(new_values))

adjusted = np.interp(img, old_values, new_values)

adjusted = adjusted.astype('uint8')

cv.imwrite('quanti.png', adjusted)
