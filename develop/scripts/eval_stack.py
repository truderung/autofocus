from pathlib import Path
import cv2 as cv
import json
from addict import Dict
import numpy as np
from matplotlib import pyplot as plt

from __internals import *

from training_data.toolbox import *



def eval_stack(exp_nr = None):
  # choose either the last out directory or the one
  # given by exp_nr (int)
  out_path = get_last_out_dir()
  if exp_nr:
    out_path = get_out_dir(exp_nr)

  # load the json file with stored informations
  stat = Dict()
  with Path(out_path, "params.json").open('r') as f:
    stat = Dict(json.loads(f.read()))


  # load heat_maps and iterate over them
  def __run_work(targets, max_value, file_ext = ""):
    hist_stack = np.empty(shape=(256,0), dtype=np.uint32)
    heat_map_stack = []

    for p in targets:
      path_stem = Path(p).stem
      with open(p, 'rb') as f:
        heat_map = np.load(f)

      heat_map = (heat_map * 255 / max_value).astype(np.uint8)
      # heat_map = clahe.apply(heat_map)

      heat_map_stack += [(path_stem, heat_map)]
      
      # colorize and store
      colored_heat_map = cv.applyColorMap(heat_map, cv.COLORMAP_JET)
      cv.imwrite(Path(out_path, f"heat_{path_stem}{file_ext}.jpg").as_posix(), colored_heat_map)

      # calculate the activation function
      hist = cv.calcHist(
        images=[heat_map],
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
        accumulate=False
      )
      hist_stack = np.hstack([hist_stack, hist])

    with open(Path(out_path, f"hist_stack{file_ext}").as_posix(), 'wb') as f:
      np.save(f, hist_stack)

    weights = exp_eval.activation(hist_stack)

    max_focus_h = 0
    max_focus_v = 0
    stack = []
    for path_stem, heat_map in heat_map_stack:
      seg_focus_h = exp_eval.exp_eval_h(heat_map, weights, seg_height=10)
      max_focus_h = np.max([max_focus_h, np.max(seg_focus_h)])
      seg_focus_v = exp_eval.exp_eval_v(heat_map, weights, seg_height=10)
      max_focus_v = np.max([max_focus_v, np.max(seg_focus_v)])
      stack += [(path_stem, heat_map, seg_focus_h, seg_focus_v)]    
    return max_focus_h, max_focus_v, stack, weights

  stacks = [
    __run_work(stat.targets, stat.max_values.max),
    __run_work(stat.targets_bar, stat.max_values_bar.max, "_bar")
  ]

  # draw
  for max_focus_h, max_focus_v, stack, _ in stacks:
    for path_stem, heat_map, seg_focus_h, seg_focus_v in stack:
      draw_heat = show.draw_h(
        img=cv.cvtColor(heat_map, cv.COLOR_GRAY2BGR),
        seg_focus=seg_focus_h,
        normalisation=max_focus_h
      )
      cv.imwrite(Path(out_path, f"draw_h_{path_stem}.jpg").as_posix(), draw_heat)
      draw_heat_v = show.draw_v(
        img=cv.cvtColor(heat_map, cv.COLOR_GRAY2BGR),
        seg_focus=seg_focus_v,
        normalisation=max_focus_v
      )
      cv.imwrite(Path(out_path, f"draw_v_{path_stem}.jpg").as_posix(), draw_heat_v)
  
  # unite
  h_s_stack = []
  v_s_stack = []
  for idx, stack in enumerate(stacks[0][2]):
    path_stem, heat_map, seg_focus_h, seg_focus_v = stack
    seg_focus_bar_h = stacks[1][2][idx][2]
    seg_focus_bar_v = stacks[1][2][idx][3]
    seg_focus_h = seg_focus_h / stacks[0][0]
    seg_focus_v = seg_focus_v / stacks[0][1]
    seg_focus_bar_h = seg_focus_bar_h / stacks[1][0]
    seg_focus_bar_v = seg_focus_bar_v / stacks[1][1]
    h_s_stack += [(path_stem, heat_map, np.amax([seg_focus_h, seg_focus_bar_h], axis=0))]
    v_s_stack += [(path_stem, heat_map, np.amax([seg_focus_v, seg_focus_bar_v], axis=0))]


  glob_seg_focus_v = np.zeros_like(v_s_stack[0][2])

  for _,_,seg_focus_v in v_s_stack:
    glob_seg_focus_v += seg_focus_v
    # mean = np.mean(seg_focus_v)
    # median = np.median(seg_focus_v)
    # std = np.std(seg_focus_v)
    # plt.figure(figsize=(15, 4))
    # plt.plot(seg_focus_v)
    # plt.axhline(median, color='m', label='Median')
    # plt.axhline(mean, color='r', label='Mean')
    # plt.axhline(mean-std, color='g', label='Mean - Std')
    # plt.axhline(median-std, color='y', label='Median - Std')
    # plt.legend()
    # print('Median= ', median)
    # print('Mean= ', mean)
    # print('Std= ', std)
    # print('Mean - Std= ', mean-std)
    # print('Median - Std= ', median-std)
    # plt.show() #block=False

  glob_seg_focus_v = glob_seg_focus_v / len(v_s_stack)
  mean = np.mean(glob_seg_focus_v)
  median = np.median(glob_seg_focus_v)
  std = np.std(glob_seg_focus_v)
  plt.figure(figsize=(15, 4))
  plt.plot(glob_seg_focus_v)
  plt.axhline(median, color='m', label='Median')
  plt.axhline(mean, color='r', label='Mean')
  plt.axhline(mean-std, color='g', label='Mean - Std')
  plt.axhline(median-std, color='y', label='Median - Std')
  plt.legend()
  print('Median= ', median)
  print('Mean= ', mean)
  print('Std= ', std)
  print('Mean - Std= ', mean-std)
  print('Median - Std= ', median-std)
  print('Max= ', np.max(glob_seg_focus_v))
  plt.show() #block=False

  max_glob_seg_focus_v = median + std  
  min_glob_seg_focus_v = median - std
  for i, focus in enumerate(glob_seg_focus_v):
    glob_seg_focus_v[i] = 1 + (median - focus)
    # if focus <= min_glob_seg_focus_v:
    #   glob_seg_focus_v[i] = 1 + 2*std
    # elif focus >= max_glob_seg_focus_v:
    #   glob_seg_focus_v[i] = 1
    # else:
    #   glob_seg_focus_v[i] = 1 + (max_glob_seg_focus_v - focus)
  
  for idx, stack in enumerate(h_s_stack):
    path_stem, heat_map, seg_focus_h = stack
    corrected_seg_focus = exp_eval.eval(heat_map, glob_weights=stacks[0][3], vertical_weights=glob_seg_focus_v, seg_height=10)
    seg_focus_v = v_s_stack[idx][2]
    draw_heat = show.draw_h(
      img=cv.cvtColor(heat_map, cv.COLOR_GRAY2BGR),
      seg_focus=seg_focus_h,
      normalisation=1,
      color=(0,255,0)
    )
    if idx > 0:
      draw_heat = show.draw_h(
        img=draw_heat,
        seg_focus=h_s_stack[idx-1][2],
        normalisation=1,
        color=(0,0,255)
      )
      draw_heat = cv.putText(
        img=draw_heat,
        text='Old',
        org=(50, 50),
        fontFace = cv.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0,0,255),
        thickness=4,
        lineType=cv.LINE_AA,
      )
      draw_heat = cv.putText(
        img=draw_heat,
        text='New',
        org=(50, 100),
        fontFace = cv.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0,255,0),
        thickness=4,
        lineType=cv.LINE_AA,
      )

    # draw_heat = show.draw_h(
    #   img=draw_heat,
    #   seg_focus=corrected_seg_focus,
    #   normalisation=max_focus_h,
    #   color=(0,255,0)
    # )
    # draw_heat = show.draw_v(
    #   img=draw_heat,
    #   seg_focus=seg_focus_v,
    #   normalisation=1,
    #   color=(255,0,0)
    # )
    cv.imwrite(Path(out_path, f"draw_{path_stem}_united.jpg").as_posix(), draw_heat)

eval_stack()

