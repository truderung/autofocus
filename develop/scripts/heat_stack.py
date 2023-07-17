from pathlib import Path
import cv2 as cv
import time
import tempfile
import re
import json
from addict import Dict
import numpy as np

from __internals import *

from training_data.toolbox import *


IMAGE_PATH = Path(WORK_PATH, 'training_data')

IMG_STACK1 = [
  Path(IMAGE_PATH, 'Stack_1/Original_images/1.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/2.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/3.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/4.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/5.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/6.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/7.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/8.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/9.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/10.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/11.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/12.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/13.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/14.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/15.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/16.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/17.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/18.jpg'),
  Path(IMAGE_PATH, 'Stack_1/Original_images/19.jpg')
]

IMG_STACK2 = [
  Path(IMAGE_PATH, 'Stack_2/Original_images/1.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/2.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/3.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/4.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/5.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/6.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/7.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/8.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/9.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/10.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/11.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/12.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/13.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/14.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/15.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/16.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/17.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/18.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/19.jpg'),
  Path(IMAGE_PATH, 'Stack_2/Original_images/20.jpg')
]

IMG_STACK3 = [
  Path(IMAGE_PATH, 'Stack_3/Original_images/01.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/02.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/03.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/04.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/05.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/06.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/07.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/08.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/09.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/10.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/11.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/12.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/13.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/14.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/15.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/16.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/17.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/18.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/19.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/20.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/21.jpg'),
  Path(IMAGE_PATH, 'Stack_3/Original_images/22.jpg')
]

IMG_STACK4 = [
  Path(IMAGE_PATH, 'Stack_4/Original_images/01.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/02.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/03.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/04.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/05.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/06.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/07.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/08.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/09.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/10.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/11.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/12.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/13.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/14.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/15.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/16.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/17.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/18.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/19.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/20.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/21.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/22.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/23.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/24.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/25.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/26.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/27.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/28.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/29.jpg'),
  Path(IMAGE_PATH, 'Stack_4/Original_images/30.jpg')
]

TEST_PROGRAMM = {
  1: {
    "gamma": list(range(40, 65)),
    "gamma_delta": 0,
    "level_distribution": [0.02],
    "width_distribution": [30]
  },
  2: {
    "gamma": [46],
    "gamma_delta": 7,
    "level_distribution": [0.005, 0.01, 0.02, 0.05, 0.1],
    "width_distribution": [30, 60, 100]
  },
  3: {
    "gamma": [46],
    "gamma_delta": 0,
    "level_distribution": [0.005, 0.02, 0.1],
    "width_distribution": [300]
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


def heat_stack():
  # plt.figure()
  # plt.plot(hist)
  # plt.show()

  def __run_work(image, half_size, mask, src_path, out_path, stat):
    heater = Heater(image, half_size)
    heat_map = heater.heat(mask)
    heat_map_bar = heater.heat_bar(mask)

    stat.std[src_path.stem] = p = float(np.std(heat_map))
    stat.mean[src_path.stem] = p = float(np.mean(heat_map))
    stat.std_bar[src_path.stem] = p = float(np.std(heat_map_bar))
    stat.mean_bar[src_path.stem] = p = float(np.mean(heat_map_bar))
        
    # store heat_maps
    heater_path = Path(out_path, f"heater_{src_path.stem}").as_posix()
    stat.targets += [heater_path]
    heater_path_bar = Path(out_path, f"heater_bar_{src_path.stem}").as_posix()
    stat.targets_bar += [heater_path_bar]
    with open(heater_path, 'wb') as f:
      np.save(f, heat_map)
    with open(heater_path_bar, 'wb') as f:
      np.save(f, heat_map_bar)

    max_value = np.max(heat_map)
    stat.max_values[src_path.stem] = max_value

    max_value_bar = np.max(heat_map_bar)
    stat.max_values_bar[src_path.stem] = max_value_bar

    return max_value, max_value_bar

  # function start
  t = time.time()
  out_path = get_new_out_dir()

  img_stack = IMG_STACK4
  prog = TEST_PROGRAMM[2]
  g = prog["gamma"][0]
  d = prog["gamma_delta"]
  gamma_range = list(range(g-d, g+d+1))
  level_distribution = prog["level_distribution"]
  width_distribution = prog["width_distribution"]
  max_value = np.float64(0.0)
  max_value_bar = np.float64(0.0)
  
  # collect debug informations
  stat = Dict()
  stat.sources = [i.as_posix() for i in img_stack]
  stat.output = out_path.as_posix()
  stat.program = prog
  stat.targets = []

  # first iteration extracted because optimization (initial processes)
  img_iter = iter(img_stack)
  src_path = next(img_iter)
  image = cv.imread(src_path.as_posix(), cv.IMREAD_GRAYSCALE)
  half_size = (image.shape[0]//2, image.shape[1]//2)

  masker = Masker(half_size)
  mask = masker.get_mask(gamma_range, level_distribution, width_distribution)
  cv.imwrite(Path(out_path, f"mask_combined.jpg").as_posix(), mask)

  # create heatmaps on two diagonals separately (gamma and gamma bar)
  max_value, max_value_bar = __run_work(image, half_size, mask, src_path, out_path, stat)

  # remained iterations
  for src_path in img_iter:
    image = cv.imread(src_path.as_posix(), cv.IMREAD_GRAYSCALE)
    new_max, new_max_bar = __run_work(image, half_size, mask, src_path, out_path, stat)
    max_value = np.max([max_value, new_max])
    max_value_bar = np.max([max_value_bar, new_max_bar])


  print(" -------------------------------------------------------- ")

  stat.max_values.max = p = float(max_value)
  stat.max_values_bar.max = p = float(max_value_bar)

  stat.timings.heating = p = time.time()-t
  print(f"heating duration: {round(p, 3)}s")

  # store json-file
  with Path(out_path, "params.json").open('w') as f:
    f.write(json.dumps(stat, indent=2))


heat_stack()

