from pathlib import Path
# import cv2 as cv

from autofocus.toolbox.masker import Masker

import pytest

# def test_mask():
#   import cv2 as cv

#   mask_image = mask(0.09, 10)
#   print(mask_image.shape)
#   # assert False
#   cv.imwrite("mask_1.jpg", mask_image[:, :, 0])

def test_create_mask():
  size = (2800, 5320)
  for g in range(0, 90):
    print(f"Go for gamma: {g}")
    masker = Masker(size)
    mask = masker.create_mask(g, 120, 200)
    # cv.imwrite(f"mask_new_{g}.jpg", mask)
