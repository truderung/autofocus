from __internals__ import setup_mask, SEG_HEIGHT, CROSSING_DISTANCE_H, setup_predict, setup_z_calibrate
import numpy as np

from autofocus.toolbox.calibrators import Predict

import pytest

##########################################################################
######    Test Predict


def test_predict_valid(setup_mask, setup_predict):
  mask = setup_mask
  data = setup_predict
  predict_flag, shift, focus_shift = Predict()(data['image_valid'], mask['mask_valid'], SEG_HEIGHT, CROSSING_DISTANCE_H, data['old_h_segment_valid'])
  assert predict_flag == "ok"
  assert type(shift) == float
  assert shift != 0
  assert np.mean(focus_shift[2]) != 1

def test_predict_unvalid_image(setup_mask, setup_predict):
  mask = setup_mask
  data = setup_predict
  predict_flag, shift, focus_shift = Predict()(data['image_unvalid'], mask['mask_valid'], SEG_HEIGHT, CROSSING_DISTANCE_H, data['old_h_segment_valid'])
  assert predict_flag == "error"
  assert type(shift) == str
  assert shift == "Image should be a 2D-array object"
  assert focus_shift == []

def test_predict_empty_image(setup_mask, setup_predict):
  mask = setup_mask
  data = setup_predict
  predict_flag, shift, focus_shift = Predict()(data['empty'], mask['mask_valid'], SEG_HEIGHT, CROSSING_DISTANCE_H, data['old_h_segment_valid'])
  assert predict_flag == "error"
  assert type(shift) == str
  assert shift == "Image should be a 2D-array object"
  assert focus_shift == []

def test_predict_valid_same_image(setup_mask, setup_predict):
  mask = setup_mask
  data = setup_predict
  predict_flag, shift, focus_shift = Predict()(data['image_same'], mask['mask_valid'], SEG_HEIGHT, CROSSING_DISTANCE_H, data['old_h_segment_valid'])
  assert predict_flag == "ok"
  assert type(shift) == float
  assert shift == 0
  assert np.mean(focus_shift[2]) == 1

def test_predict_unvalid_mask_dim(setup_mask, setup_predict):
  mask = setup_mask
  data = setup_predict
  predict_flag, shift, focus_shift = Predict()(data['image_valid'], mask['mask_unvalid_dim'], SEG_HEIGHT, CROSSING_DISTANCE_H, data['old_h_segment_valid'])
  assert predict_flag == "error"
  assert type(shift) == str
  assert shift == "Image shape or mask shape is incorrect"
  assert focus_shift == []

def test_predict_old_h_segment_empty(setup_mask, setup_predict):
  mask = setup_mask
  data = setup_predict
  predict_flag, shift, focus_shift = Predict()(data['image_valid'], mask['mask_valid'], SEG_HEIGHT, CROSSING_DISTANCE_H, data['old_h_segment_empty'])
  assert predict_flag == "error"
  assert type(shift) == str
  assert shift == "old_h_segment is not valid"
  assert focus_shift == []

def test_predict_image_too_far(setup_mask, setup_predict):
  mask = setup_mask
  data = setup_predict
  predict_flag, shift, focus_shift = Predict()(data['image_too_far'], mask['mask_valid'], SEG_HEIGHT, CROSSING_DISTANCE_H, data['old_h_segment_valid'])
  assert predict_flag == "ok"
  assert type(shift) == float
  assert abs(shift) > 1
  assert np.mean(focus_shift[2]) != 1


def test_predict_no_seg_height(setup_mask, setup_predict):
  mask = setup_mask
  data = setup_predict
  predict_flag, shift, focus_shift = Predict()(data['image_valid'], mask['mask_valid'], 0, CROSSING_DISTANCE_H, data['old_h_segment_valid'])
  assert predict_flag == "error"
  assert type(shift) == str
  assert shift == "The segment height should be a non-zero positive integer"
  assert focus_shift == []

def test_predict_no_crossing_dist(setup_mask, setup_predict):
  mask = setup_mask
  data = setup_predict
  predict_flag, shift, focus_shift = Predict()(data['image_valid'], mask['mask_valid'], SEG_HEIGHT, 0, data['old_h_segment_valid'])
  assert predict_flag == "error"
  assert type(shift) == str
  assert shift == "Invalid crossing distance"
  assert focus_shift == []


