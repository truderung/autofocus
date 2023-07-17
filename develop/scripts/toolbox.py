from numpy.linalg import inv
import cv2 as cv
import numpy as np
from numpy import linalg as la
import time

import matplotlib.pyplot as plt


class Heater:
  def __init__(self, orig_img, half_size):
    h, w = self.size = half_size
    # create dft from image, but operate only on the left top quarter of the image
    img = cv.dft(np.float64(orig_img), flags=cv.DFT_COMPLEX_OUTPUT)
    self.dft_img = img[0:h, 0:w, :]
    self.dft_img_bar = img[0:h, w:, :]

  def heat(self, mask):
    t = time.time()
    res = self.dft_img * np.dstack((mask, mask))
    print(f"dft * mask, duration: {time.time()-t}")

    t = time.time()
    res = cv.idft(res)
    print(f"idft, duration: {time.time()-t}")

    # abs of complex result
    res = cv.magnitude(res[:, :, 0], res[:, :, 1])
    # pow of 2 and norm to supress noise
    # res = res[:, :, 0]*res[:, :, 0] + res[:, :, 1]*res[:, :, 1]
    print(f"magn, duration: {time.time()-t}")

    return res

  def heat_bar(self, mask):
    t = time.time()
    mask_bar = cv.flip(mask, flipCode=1)
    res = self.dft_img_bar * np.dstack((mask_bar, mask_bar))
    print(f"dft * mask, duration: {time.time()-t}")

    t = time.time()
    res = cv.idft(res)
    print(f"idft, duration: {time.time()-t}")
    # abs of complex result
    # pow of 2 and norm to supress noise

    t = time.time()
    res = cv.magnitude(res[:, :, 0], res[:, :, 1])
    # res = res[:, :, 0]*res[:, :, 0] + res[:, :, 1]*res[:, :, 1]
    print(f"magn, duration: {time.time()-t}")

    return res


class exp_eval:
  def __init__(self) -> None:
    pass

  @staticmethod
  def exp_eval_h(img, weights, seg_height:int = 40):
    """
    @brief  Evaluates of the focus distribution along the vertical axis in an image:
    
    The image is devided in multiple horizantal segments, where each segment has a width equal to the width of the image and a height
      defined by the param. seg_height.

    @param: img:        2D-array      Input image, should be one-channel image (GRAY).
    @param: activation: int           Every grayvalue smaller than the activation, will be zero-weighted in the focus evaluation.
    @param: seg_height: int           Defines the segment height

    @return: seg_focus: 1D-array      Contains the focus evaluation for each segment.
    """

    if len(img.shape) != 2:
      raise ValueError('[ERROR]: Input image should be a one-channel image.')

    cut = img.shape[0] % seg_height
    top_cut = cut // 2
    
    nbr_segments = img.shape[0] // seg_height
    #print(img.shape, nbr_segments)
    seg_focus = np.zeros(nbr_segments, dtype=np.float64)


    # weights = list(np.zeros(activation)) + [np.exp(x/10) for x in range(255-activation)] + [0]
    # weights = np.array(weights, ndmin=2)

    # sum_weights = np.sum(weights)

    for i in range(nbr_segments):
      hist = cv.calcHist(
        images=[img[top_cut + i*seg_height : top_cut + (i + 1)*seg_height, : ]],
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
        accumulate=False
      )
      seg_focus[i] = np.dot(weights, hist)[0] #/ sum_weights
      #print(seg_focus[i])
    return seg_focus
  

  @staticmethod
  def exp_eval_v(img, weights, seg_height:int = 40):
    """
    @brief  Evaluates of the focus distribution along the horizontal axis in an image:
    
    The image is devided in multiple horizantal segments, where each segment has a width equal to the width of the image and a height
      defined by the param. seg_height.

    @param: img:        2D-array      Input image, should be one-channel image (GRAY).
    @param: activation: int           Every grayvalue smaller than the activation, will be zero-weighted in the focus evaluation.
    @param: seg_height: int           Defines the segment height

    @return: seg_focus: 1D-array      Contains the focus evaluation for each segment.
    """

    if len(img.shape) != 2:
      raise ValueError('[ERROR]: Input image should be a one-channel image.')

    cut = img.shape[1] % seg_height
    left_cut = cut // 2
    
    nbr_segments = img.shape[1] // seg_height

    seg_focus = np.zeros(nbr_segments, dtype=np.float64)

    for i in range(nbr_segments):
      hist = cv.calcHist(
        images=[img[: , left_cut + i*seg_height : left_cut + (i + 1)*seg_height]],
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
        accumulate=False
      )
      seg_focus[i] = np.dot(weights, hist)[0] #/ sum_weights
    return seg_focus

  @staticmethod
  def eval(img, glob_weights, vertical_weights, seg_height:int = 10):
    if len(img.shape) != 2:
      raise ValueError('[ERROR]: Input image should be a one-channel image.')

    cut = img.shape[0] % seg_height
    top_cut = cut // 2
    
    nbr_segments = img.shape[0] // seg_height

    seg_focus = np.zeros(nbr_segments, dtype=np.float64)

    for i in range(nbr_segments):
      for j in range(0, img.shape[1], seg_height):
        hist = cv.calcHist(
          images=[img[top_cut + i*seg_height : top_cut + (i + 1)*seg_height, j:j+seg_height]],
          channels=[0],
          mask=None,
          histSize=[256],
          ranges=[0, 256],
          accumulate=False
        )
        seg_focus[i] += np.dot(glob_weights, hist)[0] * vertical_weights[i]
    
    return seg_focus

  @staticmethod
  def sobel_eval(img, seg_height:int = 40):

    if len(img.shape) != 2:
      raise ValueError('[ERROR]: Input image should be a one-channel image.')

    cut = img.shape[0] % seg_height
    top_cut = cut // 2
    
    nbr_segments = img.shape[0] // seg_height
    seg_focus = np.zeros(nbr_segments, dtype=np.float64)

    for i in range(nbr_segments):
      # sum_seg = np.sum(
      #   img[top_cut + i*seg_height : top_cut + (i + 1)*seg_height, : ]
      # )
      hist = cv.calcHist(
        images=[img[top_cut + i*seg_height : top_cut + (i + 1)*seg_height, : ]],
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
        accumulate=False
      )
      seg_focus[i] = hist[255]

    return seg_focus

  @staticmethod
  def laplace_eval(img, seg_height:int = 40):

    cut = img.shape[0] % seg_height
    top_cut = cut // 2
    
    nbr_segments = img.shape[0] // seg_height
    seg_focus = np.zeros(nbr_segments, dtype=np.float64)

    for i in range(nbr_segments):
      seg_focus[i] = np.sum(
        img[top_cut + i*seg_height : top_cut + (i + 1)*seg_height, : ],
        axis=None
      )
    
    return seg_focus

  @staticmethod
  def activation_Sobel(otsu_threshold):

    s = 1 / (255 - otsu_threshold)

    def __Sigmoid(x, threshold):
      if x < threshold:
        return 0
      return (x-threshold) * s

    filt_1 = np.array([__Sigmoid(x, otsu_threshold) for x in range(0, 256)], dtype=np.float64)


    return filt_1

  @staticmethod
  def activation(hist_stack, otsu_threshold):

    if otsu_threshold:
      threshold = otsu_threshold
    else:
      hists = hist_stack.cumsum(1)[:, -1]
      hist_mean = np.mean(hists)
      hist_std = np.std(hists)
      
      filt = hists[hists < hist_mean+hist_std]
      threshold = len(hists) - len(filt)

    s = 1 / (255 - threshold)

    def __Sigmoid(x, threshold):
      # x_ = x-threshold
      # return x_ / (1 + abs(x_)) + 1
      if x < threshold:
        return 0
      return (x-threshold) * s


    filt_1 = np.array([__Sigmoid(x, threshold) for x in range(0, 256)], dtype=np.float64)
    
    return filt_1

  # @staticmethod
  # def activation(stack_hist, half_size, percent, new):
  #   """
  #   @brief  Calculates the (activation) threshold grayvalue.

  #   @param: stack_hist:     2D-array    Histogramms of the stack images.
  #   @param: half_size       (int, int)  Half-Height and Half-width of the original image size.
  #   @param: percent         Float       Percentage of pixels to be thresholded.

  #   @return: activation     int         Threshold Grayvalue.
  #   """
  #   accum_hist = np.zeros((256,1))

  #   for hist in stack_hist:
  #     accum_hist = np.add(accum_hist, hist)
    
  #   for i in range(1, len(accum_hist)):
  #     accum_hist[i] = accum_hist[i] + accum_hist[i-1]

  #     if accum_hist[i] >= percent*half_size[0]*half_size[1]*len(stack_hist):
  #       return i
    
  #   return i



class Show:
  def __init__(self) -> None:
    pass

  @staticmethod
  def colorize_heat_map(img, max_val = 3*256-1):
    img = (img * 255 / max_val).astype(np.uint8)
    #img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)

    heat_map = cv.applyColorMap(img, cv.COLORMAP_JET)
    return heat_map
  
  @staticmethod
  def plot(segs_focus):
    """
    @brief   Uses Matplotlib to plot the focus graphs.

    @param: segs_focus:     1D-array or 2D-array     Input array.
      If the Input array is a 1D-array, one graph will be ploted
      If the Input array is a 2D-array, all graphs will be ploted in one figure.

    @return: None
    """
    plt.figure()

    nbr_segs_focus = len(segs_focus)
    x_axis = np.arange(0, segs_focus[0].shape[0])

    for i in range(nbr_segs_focus):
      plt.plot(x_axis, segs_focus[i], label=f"{i+1}")

    plt.legend(title='Focus distribution in the image Nbr.:')

    plt.grid(axis='x', color='0.95')
    plt.xlabel('Segments')
    plt.ylabel('Focus distribution')

    plt.show()


  @staticmethod
  def draw_h(img, seg_focus, normalisation, color=(0,0,255), title='', pos=(0,0)):

    """
    @brief  Draws a graph for the focus distribution along the vertical axis of the Image.

    @param: img:        2D-array        Input image, should be a three-channel image (BGR).
    @param: seg_focus:  1D-array        Contains the focus evaluation for each segment.
    @param: normalisation: float||Bool  For a stack of Images, normalisation should be a float number equal to the max focus in the Stack.
      If the number of Images is equal to 1, normalisation should be equal to False.

    @return: img:       2D-array        Output image, with the focus distribution drawn on the left side.
    """

    if len(img.shape) != 3:
      raise ValueError('[ERROR]: Input image should be a three-channel image.')

    seg_height = img.shape[0] // len(seg_focus)

    top_cut = (img.shape[0] % seg_height) // 2
    max_curve_height = 500

    if normalisation:
      [min_focus, max_focus] = normalisation
    else:
      max_focus = max(seg_focus)
      min_focus = min(seg_focus)

    delta_focus = max_focus - min_focus

    seg_focus = (((seg_focus-min_focus)/delta_focus)*max_curve_height).astype(np.uint16)

    for i in range(len(seg_focus)-1):
      img = cv.line(
        img=img,
        pt1=(100+seg_focus[i], top_cut+seg_height//2+i*seg_height),
        pt2=(100+seg_focus[i+1], top_cut+seg_height//2+(i+1)*seg_height),
        color=color,
        thickness=4,
        lineType=cv.LINE_8,
      )
    
    cv.putText(
      img=img,
      text=title,
      org=pos,
      fontFace=cv.FONT_HERSHEY_SIMPLEX,
      fontScale=1,
      color=color,
      thickness=2
    )

    return img

  @staticmethod
  def draw_v(img, seg_focus, normalisation, color=(0,0,255)):

    """
    @brief  Draws a graph for the focus distribution along the horizontal axis of the Image.

    @param: img:        2D-array        Input image, should be a three-channel image (BGR).
    @param: seg_focus:  1D-array        Contains the focus evaluation for each segment.
    @param: normalisation: float||Bool  For a stack of Images, normalisation should be a float number equal to the max focus in the Stack.
      If the number of Images is equal to 1, normalisation should be equal to False.

    @return: img:       2D-array        Output image, with the focus distribution drawn on the left side.
    """

    if len(img.shape) != 3:
      raise ValueError('[ERROR]: Input image should be a three-channel image.')

    seg_height = img.shape[1] // len(seg_focus)

    left_cut = (img.shape[1] % seg_height) // 2
    max_curve_height = 500

    if normalisation:
      max_focus = normalisation
    else:
      max_focus = max(seg_focus)

    seg_focus = ((seg_focus/max_focus)*max_curve_height).astype(np.uint16)

    for i in range(len(seg_focus)-1):
      img = cv.line(
        img=img,
        pt1=(left_cut+seg_height//2+i*seg_height, 100+seg_focus[i]),
        pt2=(left_cut+seg_height//2+(i+1)*seg_height, 100+seg_focus[i+1]),
        color=color,
        thickness=4,
        lineType=cv.LINE_8,
      )

    return img


def Kalman(signal):
  X = np.array([
    [signal[0]],
    [0],
    [0],
    [0]
  ])
  
  P = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1000, 0],
    [0, 0, 0, 1000]
  ])

  A = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ])

  H = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0]
  ])

  I = np.identity(4)

  R = np.array([
    [0.0225, 0],
    [0, 0.0225]
  ])

  noise_ax = 250
  noise_ay = 1

  Q = np.array([
    [0.25*noise_ax, 0, 0.5*noise_ax, 0],
    [0, 0.25*noise_ay, 0, 0.5*noise_ay],
    [0.5*noise_ax, 0, noise_ax, 0],
    [0.5*noise_ay, 0, noise_ay, 0]
  ])

  z = np.zeros([2, 1])
  new_signal = []

  for i in range (len(signal)):
    z[0][0] = signal[i]
    z[1][0] = i

    #Predict
    X = np.matmul(A, X)
    At = np.transpose(A)
    P = np.add(np.matmul(A, np.matmul(P, At)), Q)

    # Measurement update step
    Y = np.subtract(z, np.matmul(H, X))
    Ht = np.transpose(H)
    S = np.add(np.matmul(H, np.matmul(P, Ht)), R)
    K = np.matmul(P, Ht)
    Si = inv(S)
    K = np.matmul(K, Si)
    
    # New state
    X = np.add(X, np.matmul(K, Y))
    P = np.matmul(np.subtract(I ,np.matmul(K, H)), P)

    new_signal.append(X[0][0])
  
  return new_signal

