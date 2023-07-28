from autofocus.toolbox.calibrators import *

import pytest

##########################################################################
######    Test Constractors


def test_calibrateY_construction():
  assert type(CalibrateY()) == CalibrateY

def test_calibrateZ_construction():
  assert type(CalibrateZ()) == CalibrateZ

def test_Predict_construction():
  assert type(Predict()) == Predict


