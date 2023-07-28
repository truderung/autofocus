from __internals__ import setup_mask, SEG_HEIGHT, CROSSING_DISTANCE_H, setup_y_calibrate
import numpy as np

from autofocus.toolbox.calibrators import CalibrateY

import pytest

##########################################################################
######    Test Y-Calibration


def test_y_calibration_valid(setup_mask, setup_y_calibrate):
  mask = setup_mask
  data = setup_y_calibrate
  y_flag, y_correction, all_indexes = CalibrateY()(data['y_images_valid'], mask['mask_valid'], SEG_HEIGHT, CROSSING_DISTANCE_H)
  assert y_flag == "ok"
  assert type(y_correction) == float
  assert len(all_indexes) != 0

def test_y_calibration_unvalid_images(setup_mask, setup_y_calibrate):
  mask = setup_mask
  data = setup_y_calibrate
  y_flag, y_correction, all_indexes = CalibrateY()(data['y_images_unvalid'], mask['mask_valid'], SEG_HEIGHT, CROSSING_DISTANCE_H)
  assert y_flag == "error"
  assert type(y_correction) == str
  assert len(all_indexes) != 0

def test_y_calibration_no_images(setup_mask, setup_y_calibrate):
  mask = setup_mask
  data = setup_y_calibrate
  y_flag, y_correction, all_indexes = CalibrateY()(data['empty'], mask['mask_valid'], SEG_HEIGHT, CROSSING_DISTANCE_H)
  assert y_flag == "error"
  assert type(y_correction) == str
  assert y_correction == "No images provided"
  assert all_indexes == []

def test_y_calibration_unvalid_mask_dim(setup_mask, setup_y_calibrate):
  mask = setup_mask
  data = setup_y_calibrate
  y_flag, y_correction, all_indexes = CalibrateY()(data['y_images_valid'], mask['mask_unvalid_dim'][0], SEG_HEIGHT, CROSSING_DISTANCE_H)
  assert y_flag == "error"
  assert type(y_correction) == str
  assert y_correction == "Image shape or mask shape is incorrect"
  assert all_indexes == []

def test_y_calibration_no_seg_height(setup_mask, setup_y_calibrate):
  mask = setup_mask
  data = setup_y_calibrate
  y_flag, y_correction, all_indexes = CalibrateY()(data['y_images_valid'], mask['mask_valid'], 0, CROSSING_DISTANCE_H)
  assert y_flag == "error"
  assert type(y_correction) == str
  assert y_correction == "The segment height should be a non-zero positive integer"
  assert all_indexes == []

def test_y_calibration_no_crossing_dist(setup_mask, setup_y_calibrate):
  mask = setup_mask
  data = setup_y_calibrate
  y_flag, y_correction, all_indexes = CalibrateY()(data['y_images_valid'], mask['mask_valid'], SEG_HEIGHT, 0)
  assert y_flag == "error"
  assert type(y_correction) == str
  assert y_correction == "Invalid crossing distance"
  assert all_indexes == []
