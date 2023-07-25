from utils import *
from masker import Masker
from pathlib import Path

IMAGE_PATH = Path('/home/issam/Desktop/Github/auto_focusing/training_data')

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
  Path(IMAGE_PATH, 'Stack_5/Original_images/51.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/50.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/49.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/48.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/47.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/46.jpg'),

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

IMG_STACK5 = [
  Path(IMAGE_PATH, 'Stack_5/Original_images/45.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/44.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/43.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/42.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/41.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/40.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/39.jpg'),
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
  Path(IMAGE_PATH, 'Stack_5/Original_images/26.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/25.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/24.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/23.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/22.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/21.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/20.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/19.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/18.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/17.jpg')
]


g = 46
d = 7
gamma_range = list(range(g-d, g+d+1))
level_distribution = [0.005, 0.01, 0.02, 0.05, 0.1]
width_distribution = [30, 60, 100]


masker = Masker((1400, 2660))
mask = masker.get_mask(gamma_range, level_distribution, width_distribution)
print("Mask created..................")


max_value = max_value_bar = 0
heat_maps = []
heat_maps_bar = []

print("Start Loading images..........")

for img_path in IMG_STACK5:
  image = cv.imread(img_path.as_posix(), cv.IMREAD_GRAYSCALE)
  res, res_bar = heat_double(image, mask)
  max_value = max(max_value, np.max(res))
  max_value_bar = max(max_value_bar, np.max(res_bar))
  heat_maps.append(res)
  heat_maps_bar.append(res_bar)

print("Images were heated............")


h_segments_stack = eval_stack(heat_maps, max_value, heat_maps_bar, max_value_bar, 10)

'''for i in range(len(h_segments_stack)):
  if max(h_segments_stack[i])==1:
    max_seg_index = h_segments_stack[i].argmax()

for i in range(len(h_segments_stack)):
  print(f"At Image {i}, Focus segment {max_seg_index} gleich {h_segments_stack[i][max_seg_index]}")'''

y_flag, y_correction, units = calibrate_y(h_segments_stack)

if y_flag == False:
  print(f"Move Up by {y_correction} and take {units} images")

else:
  dist = y_correction * 0.00274 * 10
  print(f"Y_Correction: {dist}")
  best = evaluate_roi((220, 1180), y_correction, h_segments_stack, 10)
  print("Best position:", best)

'''for seg in h_segments_stack[best]:
  print(f"Segment focus: {seg}")

max_best = h_segments_stack[best].argmax()

for seg in h_segments_stack[best]:
  seg_n = seg/h_segments_stack[best][max_best]
  print(f"Nomm. seg focus: {seg_n}")'''


shift = predict(h_segments_stack[best], h_segments_stack[best], 40)

print(f"Shift = {shift}")