from __internals__ import setup_mask, SEG_HEIGHT, CROSSING_DISTANCE_H, setup_z_calibrate
import numpy as np

from autofocus.toolbox.calibrators import CalibrateZ

import pytest

##########################################################################
######    Test Z-Calibration


def test_z_calibration_valid(setup_mask, setup_z_calibrate):
  mask = setup_mask
  data = setup_z_calibrate
  z_flag, z_correction, old_h_segment = CalibrateZ()(data['z_images_valid'], mask['mask_valid'], (350,1050), 0, SEG_HEIGHT)
  assert type(z_correction) == np.int64
  assert z_correction >=0 and z_correction <= len(data['z_images_valid'])
  assert len(old_h_segment[2]) == 1400//SEG_HEIGHT
  assert type(old_h_segment[0][0]) == np.float64
  assert type(old_h_segment[1][0]) == np.float64
  assert z_flag == "ok"

def test_z_calibration_no_images(setup_mask, setup_z_calibrate):
  mask = setup_mask
  data = setup_z_calibrate
  z_flag, z_correction, old_h_segment = CalibrateZ()(data['empty'], mask['mask_valid'], (350,1050), 0, SEG_HEIGHT)
  assert z_flag == "error"
  assert type(z_correction) == str
  assert z_correction == "No images provided"
  assert old_h_segment == []

def test_z_calibration_unvalid_mask_dim(setup_mask, setup_z_calibrate):
  mask = setup_mask
  data = setup_z_calibrate
  z_flag, z_correction, old_h_segment = CalibrateZ()(data['z_images_valid'], mask['mask_unvalid_dim'][0], (350,1050), 0, SEG_HEIGHT)
  assert z_flag == "error"
  assert type(z_correction) == str
  assert z_correction == "Image shape or mask shape is incorrect"
  assert old_h_segment == []

def test_z_calibration_no_roi(setup_mask, setup_z_calibrate):
  mask = setup_mask
  data = setup_z_calibrate
  z_flag, z_correction, old_h_segment = CalibrateZ()(data['z_images_valid'], mask['mask_valid'], (0,0), 0, SEG_HEIGHT)
  assert z_flag == "error"
  assert type(z_correction) == str
  assert z_correction == "Region of interest should be at least equal to one segment height"
  assert old_h_segment == []