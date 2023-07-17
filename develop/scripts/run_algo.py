from algo import Stack

from pathlib import Path

from __internals import *

from autofocus.toolbox import *


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



stack_1 = Stack(img_stack=IMG_STACK1, seg_height=10)
stack_1.heat_stack(TEST_PROGRAMM[2])
stack_1.eval_stack()
stack_1.sobel_mask()
stack_1.laplace()
stack_1.draw(DFT=True ,SOBEL=True ,SOBEL_THRES=True, LAPLACE=True)
stack_1.evaluate()