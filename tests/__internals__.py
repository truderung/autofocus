from autofocus.toolbox.calibrators import *
from autofocus.toolbox.masker import Masker
from pathlib import Path
import cv2 as cv

import pytest


WORK_PATH = Path(__file__).parent.parent
IMAGE_PATH = Path(WORK_PATH, 'develop/data')


'''
  Path(IMAGE_PATH, 'Stack_5/Original_images/68.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/67.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/66.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/65.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/64.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/63.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/62.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/61.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/60.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/59.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/58.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/57.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/56.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/55.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/54.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/53.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/52.jpg'),


  Path(IMAGE_PATH, 'Stack_5/Original_images/25.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/24.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/23.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/22.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/21.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/20.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/19.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/18.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/17.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/16.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/15.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/14.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/13.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/12.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/11.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/10.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/09.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/08.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/07.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/06.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/05.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/04.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/03.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/02.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/01.jpg') 
'''

IMG_STACK_Y_CALIB = [
  Path(IMAGE_PATH, 'Stack_5/Original_images/51.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/50.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/49.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/48.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/47.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/46.jpg'),  
  Path(IMAGE_PATH, 'Stack_5/Original_images/45.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/44.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/43.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/42.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/41.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/40.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/39.jpg')
]

IMG_STACK_Y_CALIB_UNVALID = [
  Path(IMAGE_PATH, 'Stack_5/Original_images/19.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/39.jpg')
]


IMG_STACK_Z_CALIB = [
  Path(IMAGE_PATH, 'Stack_5/Original_images/38.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/37.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/36.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/35.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/34.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/33.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/32.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/31.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/30.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/29.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/28.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/27.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/26.jpg')
]


SEG_HEIGHT = 10
CROSSING_DISTANCE_H = 777


@pytest.fixture(scope='module', autouse=True)
def setup_mask():
  # tear_up

  g = 46
  d = 7
  gamma_range = list(range(g-d, g+d+1))
  level_distribution = [0.005, 0.01, 0.02, 0.05, 0.1]
  width_distribution = [30, 60, 100]

  masker_valid = Masker((1400, 2660))
  mask_valid = masker_valid.get_mask(gamma_range, level_distribution, width_distribution)
  
  masker_unvalid_dim = Masker((2800, 5320))
  mask_unvalid_dim = masker_unvalid_dim.get_mask(gamma_range, level_distribution, width_distribution)
  
  
  mask = dict()

  mask["mask_valid"] = mask_valid
  mask["mask_unvalid_dim"] = mask_unvalid_dim

  yield mask

  # tear_down
  pass



@pytest.fixture(scope='module', autouse=True)
def setup_y_calibrate():
  y_images_valid = []
  for p in IMG_STACK_Y_CALIB:
    img = cv.imread(p.as_posix(), cv.IMREAD_GRAYSCALE)
    y_images_valid.append(img)

  y_images_unvalid = []
  for p in IMG_STACK_Y_CALIB_UNVALID:
    img = cv.imread(p.as_posix(), cv.IMREAD_GRAYSCALE)
    y_images_unvalid.append(img)

  y_calibrate = dict()

  y_calibrate["y_images_valid"] = y_images_valid
  y_calibrate["y_images_unvalid"] = y_images_unvalid
  y_calibrate["empty"] = []

  yield y_calibrate
  # tear_down
  pass


@pytest.fixture(scope='module', autouse=True)
def setup_z_calibrate():
  z_images_valid = []
  for p in IMG_STACK_Z_CALIB:
    img = cv.imread(p.as_posix(), cv.IMREAD_GRAYSCALE)
    z_images_valid.append(img)

  z_calibrate = dict()

  z_calibrate["z_images_valid"] = z_images_valid
  z_calibrate["one_image"] = [z_images_valid[3]]
  z_calibrate["empty"] = []

  yield z_calibrate
  # tear_down
  pass



@pytest.fixture(scope='module', autouse=True)
def setup_predict(setup_mask, setup_z_calibrate):
  mask = setup_mask
  z_images = setup_z_calibrate

  _, z_correction, old_h_segment = CalibrateZ()(z_images['z_images_valid'], mask['mask_valid'], (350,1050), 0, SEG_HEIGHT)

  predict = dict()
  predict["old_h_segment_valid"] = old_h_segment
  predict["old_h_segment_empty"] = []

  predict["image_valid"] = z_images['z_images_valid'][z_correction-1]
  predict["image_unvalid"] = 16
  predict["image_same"] = z_images['z_images_valid'][z_correction]
  predict["image_too_far"] = cv.imread(Path(IMAGE_PATH, 'Stack_5/Original_images/65.jpg').as_posix(), cv.IMREAD_GRAYSCALE)
  predict["empty"] = []

  yield predict
 
  # tear_down
  pass


'''  g = 46
  d = 7
  gamma_range = list(range(g-d, g+d+1))
  level_distribution = [0.005, 0.01, 0.02, 0.05, 0.1]
  width_distribution = [30, 60, 100]

  masker = Masker((1400, 2660))
  mask = masker.get_mask(gamma_range, level_distribution, width_distribution)'''