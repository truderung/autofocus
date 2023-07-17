import time
import json
import numpy as np
import cv2 as cv

from pathlib import Path
from addict import Dict

from autofocus.toolbox.masker import Masker

from __internals import *
from toolbox import *


IMAGES = {
  #Path(WORK_PATH, '18_57_quad.jpg').as_posix(): [3]
  # Path(WORK_PATH, '18_57.jpg').as_posix(): [3]
  # Path(WORK_PATH, 'data/Stack_1/Original_images/5.jpg').as_posix(): [3],
  Path(WORK_PATH, 'data/Stack_1/Original_images/10.jpg').as_posix(): [3],
  # Path(WORK_PATH, 'data/Stack_1/Original_images/12.jpg').as_posix(): [3]
  # Path(WORK_PATH, 'data/Stack_1/Original_images/16.jpg').as_posix(): [3],
}

TEST_PROGRAMM = {
  1: {
    "gamma": list(range(40, 65)),
    "gamma_delta": 0,
    "level_distribution": [0.02],
    "width_distribution": [30]
  },
  2: {
    "gamma": [46],
    "gamma_delta": 10,
    "level_distribution": [0.005, 0.01, 0.02, 0.05, 0.1],
    "width_distribution": [30, 60, 100]
  },
  3: {
    "gamma": [46],
    "gamma_delta": 7,
    "level_distribution": [0.005, 0.02, 0.1],
    "width_distribution": [40, 100]
  }, 
  4: {
    "gamma": [20],
    "gamma_delta": 1,
    "level_distribution": [0, 0.01], #[0, 0.01, 0.02, 0.05, 0.1],
    "width_distribution": [10, 30] # [10, 30, 50, 100]
  },
  5: {
    "gamma": [57],
    "gamma_delta": 2,
    "level_distribution": [0.05, 0.1],
    "width_distribution": [30] # [10, 30, 50, 100]
  }  
}



def test_heat(verbose = False):
  for src_path, prog_list in IMAGES.items():
    image = cv.imread(src_path, cv.IMREAD_GRAYSCALE)
    half_size = (image.shape[0]//2, image.shape[1]//2)
    masker = Masker(half_size)

    for i in prog_list:
      out_path = get_new_out_dir()
      stat = Dict()

      prog = TEST_PROGRAMM[i]
      max_value = np.float64(0.0)
      
      # heat and store
      for g in prog["gamma"]:
        d = prog["gamma_delta"]
        gamma_range = list(range(g-d, g+d+1))
        level_distribution = prog["level_distribution"]
        width_distribution = prog["width_distribution"]

        t = time.time()
        mask = masker.get_mask(gamma_range, level_distribution, width_distribution)
        stat.timings.mask_creation[f"{g}"] = p = time.time()-t
        print(f"mask creation for gamma {g}, duration: {p}")
    
        if verbose:
          cv.imwrite(Path(out_path, f"mask_g{g}.jpg").as_posix(), mask)

        heater = Heater(image, half_size)

        t = time.time()
        heat_map = heater.heat(mask)
        stat.variance[f"{g}"] = p = float(np.std(heat_map))
        print(f"variance for gamma {g}: {p}")

        max_value = np.max([max_value, np.max(heat_map)])

        # store heat_maps
        with open(Path(out_path, f"heater_{g}").as_posix(), 'wb') as f:
          np.save(f, heat_map)
        stat.timings.heat_and_save[f"{g}"] = p = time.time()-t
        print(f"heat and save for gamma {g}, duration: {time.time()-t}")

      # max_value = 974266512052568
      stat.max_value = p = float(max_value)
      print(f"max_value: {p}")
   
      # collect all statistics
      stat.source = src_path
      stat.output = out_path.as_posix()
      stat.program = prog

      # load and color map
      for g in prog["gamma"]:
        t = time.time()
        with open(Path(out_path, f"heater_{g}").as_posix(), 'rb') as f:
          heat_map = np.load(f)      
        heat_map = Show.colorize_heat_map(heat_map, max_value)
        stat.timings.load_and_map[f"{g}"] = p = time.time()-t
        print(f"load and map for gamma {g}, duration: {p}")
        cv.imwrite(Path(out_path, f"heat_{g}.jpg").as_posix(), heat_map)

      with Path(out_path, "params.json").open('w') as f:
        f.write(json.dumps(stat, indent=2))

    print(" -------------------------------------------------------- ")


test_heat(True)

