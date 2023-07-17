from numpy.linalg import inv
import cv2 as cv
import numpy as np
from numpy import linalg as la

PI_180 = np.pi/180.0
_180_PI = 180/np.pi


class Masker:
  def __init__(self, size):
    self.size = size
    # the major axis of the ellipse is twice as 
    # big as the diagonal d of the mask size
    self.diagonal = 2 * la.norm([size[0], size[1]])

  def get_mask(self, gamma_distribution = [45.0], level_distribution = [0.05, 0.11, 0.17, 0.22], width_distribution = [100]):
    mask = np.zeros(self.size, dtype=np.uint16)

    for g in gamma_distribution:
      for l in list(np.array(level_distribution)*self.diagonal):
        l = round(l)
        for w in width_distribution:
          mask += self.create_mask(g, l, w)

    mask = mask / np.max(mask)
    return mask

  """
  Create a filter mask

  # gamma: braiding angle of stent
  # level: offset in pixel (euklidean)
  # width: minor axis of the ellipse, Öffnung der semi-parabel 
  """
  def create_mask(self, gamma: float, level: int, width: int):
    mask_height, mask_width = self.size
    d = self.diagonal

    # we round the angle here because cv.ellipse rounds inside the function
    gamma_bar = round(np.arctan(np.tan(gamma * PI_180) * mask_width/mask_height) * _180_PI)
    x_shift = round(np.sin(gamma_bar * PI_180)*d)
    y_shift = round(np.cos(gamma_bar * PI_180)*d)

    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    cv.ellipse(mask, (x_shift, y_shift), (width, round(d-2*level)), -gamma_bar, 180, 360, 255, -1)

    return mask
