import cv2 as cv
import numpy as np
from statistics import mean, median
from abc import ABC, abstractmethod


class Calibrate(ABC):
  '''
  @class Calibrate
  Abstract base class for calibration and image processing.

  This class defines the basic structure for calibration and provides common methods for focus evaluation.

  @attribute None

  @methods
  - activation(self, hist_stack):
      Calculates the activation function for histogram-based image thresholding.

  - horz_eval(self, img, weights, seg_height: int = 10):
      Evaluates the focus distribution along the vertical axis in an image.

  - heat_double(self, orig_img, mask):
      Heats the image in two halves using Discrete Fourier Transform (DFT).

  - eval_stack(self, images, mask, segment_height):
      Evaluates the stack of heat maps and computes focus evaluation for each segment.

  @usage
  ```
  # Create a subclass inheriting from Calibrate and implement abstract methods.
  class MyCalibrate(Calibrate):
      # Implement abstract methods here.

  # Create an instance of the subclass
  calibrator = MyCalibrate()

  # Call the methods
  activation_weights = calibrator.activation(hist_stack)
  focus_distribution = calibrator.horz_eval(image, activation_weights, segment_height)
  heated_image, heated_image_bar = calibrator.heat_double(orig_img, mask)
  focus_evaluation = calibrator.eval_stack(images, mask, segment_height)
  ```
  '''

  def activation(self, hist_stack):
    """
    @brief Calculates the activation function for histogram-based image thresholding.

    The activation function is computed based on the input histogram stack using the mean and standard deviation of the histogram stack.

    @param hist_stack:    2D-array    Histogram stack containing cumulative histograms for multiple images.

    @return filt_1:     1D-array    Activation function values for each gray level.
    """

    hists = hist_stack.cumsum(1)[:, -1]
    hist_mean = np.mean(hists)
    hist_std = np.std(hists)

    filt = hists[hists < hist_mean + hist_std]
    threshold = len(hists) - len(filt)

    s = 1 / (255 - threshold)

    def __run_work(x: int):
      """
      @brief __runwork function used for calculating the activation value.

      This function calculates the activation value based on the input x and threshold.

      @param x:     int     Input value (gray level).
      @param threshold: int     Activation threshold.

      @return:      float   Activation value.
      """
      if x < threshold:
        return 0
      return (x - threshold) * s

    weights = np.array([__run_work(x) for x in range(0, 256)], dtype=np.float64)

    return weights


  def horz_eval(self, img, weights, seg_height: int = 10):
    """
    @brief Evaluates the focus distribution along the vertical axis in an image.

    The image is divided into multiple horizontal segments, where each segment has the same width as the image and a height defined by the `seg_height` parameter.

    @param img:    2D-array    Input image. Should be a one-channel grayscale image.
    @param weights:  1D-array    Weight values used for focus evaluation. Determines the contribution of different gray values.
    @param seg_height: int       Defines the segment height. Defaults to 40.

    @throws ValueError: If the input image has more than one channel.

    @return seg_focus: 1D-array    Contains the focus evaluation for each segment.
    """

    if len(img.shape) != 2:
      raise ValueError("[ERROR]: Input image should be a one-channel image.")

    cut = img.shape[0] % seg_height
    top_cut = cut // 2

    nbr_segments = img.shape[0] // seg_height
    seg_focus = np.zeros(nbr_segments, dtype=np.float64)

    for i in range(nbr_segments):
      hist = cv.calcHist(
        images=[img[top_cut + i * seg_height : top_cut + (i + 1) * seg_height, :]],
        channels=[0],
        mask=None,
        histSize=[256],
        ranges=[0, 256],
        accumulate=False,
      )
      seg_focus[i] = np.dot(weights, hist)[0]

    return seg_focus


  def heat_double(self, orig_img, mask):
    """
    @brief Heats the image in two halves using Discrete Fourier Transform (DFT).

    This function divides the original image into two halves and applies the heating process to each half separately.
    It computes the DFT of the image, restricts it to the left and right halves, and applies the corresponding masks.
    The resulting heated images for each half are returned.

    @param orig_img: 2D-array    Original image.
    @param mask:   2D-array    Mask to be applied during the heating process.

    @return: res:    2D-array   Resulting heated image for the left half.
    @return: res_bar:  2D-array   Resulting heated image for the right half.
    """

    def __heat(dft_img, mask):
      """
      @brief Heats the image using Discrete Fourier Transform (DFT).

      This function applies the provided mask to the DFT image, performs inverse DFT, and calculates the magnitude.

      @param dft_img: 3D-array    DFT of the original image.
      @param mask:  2D-array    Mask to be applied during the heating process.

      @return: res: 2D-array    Resulting heated image.
      """

      res = dft_img * np.dstack((mask, mask))
      res = cv.idft(res)
      res = cv.magnitude(res[:, :, 0], res[:, :, 1])

      return res


    h, w = orig_img.shape[0] // 2, orig_img.shape[1] // 2
    img = cv.dft(np.float64(orig_img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_img = img[0:h, 0:w, :]
    dft_img_bar = img[0:h, w:, :]
    mask_bar = cv.flip(mask, flipCode=1)

    res = __heat(dft_img, mask)
    res_bar = __heat(dft_img_bar, mask_bar)

    return res, res_bar
  

  def eval_stack(self, images, mask, segment_height):
    """
    @brief Evaluates the stack of heat maps and computes focus evaluation for each segment.

    This function takes two stacks of heat maps, their corresponding maximum values, and the segment height as input.

    It calculates the histograms for each heat map, computes the activation weights,
      and evaluates the focus distribution for each segment using the `horz_eval` function.

    @param heat_maps:     List of 2D-arrays   Stack of heat maps.
    @param max_value:     float         Maximum value of the heat maps.
    @param heat_maps_bar:   List of 2D-arrays   Stack of heat maps (bar).
    @param max_value_bar:   float         Maximum value of the heat maps (bar).
    @param segment_height:  int         Height of each segment.

    @return h_s_stack:    List of 1D-arrays   Focus evaluation for each segment.
    """

    max_value = 0
    max_value_bar = 0
    heat_maps = []
    heat_maps_bar = []

    for image in images:
      res, res_bar = self.heat_double(image, mask)
      max_value = max(max_value, np.max(res))
      max_value_bar = max(max_value_bar, np.max(res_bar))
      heat_maps.append(res)
      heat_maps_bar.append(res_bar)


    def __run_work(heat_maps, max_value):
      """
      @brief Helper function to compute focus evaluation for a stack of heat maps.

      This function calculates the histograms for each heat map, computes the activation weights using the `activation` function,
      and evaluates the focus distribution for each segment using the `horz_eval` function.

      @param heat_maps:       List of 2D-arrays   Stack of heat maps.
      @param max_value:       float         Maximum value of the heat maps.

      @return max_focus:      float         Maximum focus value.
      @return h_segments_stack:   List of 1D-arrays   Focus evaluation for each segment.
      """
      hist_stack = np.empty(shape=(256, 0), dtype=np.uint32)
      heat_map_stack = []

      for heat_map in heat_maps:
        heat_map = (heat_map * 255 / max_value).astype(np.uint8)

        heat_map_stack += [heat_map]

        hist = cv.calcHist(
          images=[heat_map],
          channels=[0],
          mask=None,
          histSize=[256],
          ranges=[0, 256],
          accumulate=False,
        )
        hist_stack = np.hstack([hist_stack, hist])

      weights = self.activation(hist_stack)

      max_focus = 0
      h_segments_stack = []

      for heat_map in heat_map_stack:
        seg_focus = self.horz_eval(heat_map, weights, segment_height)
        max_focus = np.max([max_focus, np.max(seg_focus)])
        h_segments_stack += [seg_focus]

      return max_focus, h_segments_stack

    h_segments_stack_double = [
      __run_work(heat_maps, max_value),
      __run_work(heat_maps_bar, max_value_bar),
    ]

    h_segments_stack_united = []
    for idx, seg_focus in enumerate(h_segments_stack_double[0][1]):
      seg_focus_bar = h_segments_stack_double[1][1][idx]
      seg_focus = seg_focus / h_segments_stack_double[0][0]
      seg_focus_bar = seg_focus_bar / h_segments_stack_double[1][0]
      h_segments_stack_united += [np.amax([seg_focus, seg_focus_bar], axis=0)]
    
    return h_segments_stack_united



class CalibrateY(Calibrate):
    '''
    @class CalibrateY
    Class to perform Y calibration.

    @brief This class inherits from the `Calibrate` class and implements the Y calibration
    process for a given set of images.

    @attribute None

    @methods
    - __call__(self, images, mask, segment_height, region):
        Perform Y calibration for the given images.

    - __calibrate_y(self, images, mask, segment_height, region):
        Internal method to perform Y calibration for the given images.

    @usage
    ```
    y_flag, y_correction, all_indexes = CalibrateY()(images, mask, segment_height, region)
    ```
    '''

    def __call__(self, images, mask, segment_height, region):
        '''
        @brief Perform Y calibration for the given images.

        @param images (list): List of images to perform Y calibration on.
        @param mask (numpy.ndarray): The DFT mask used for image processing.
        @param segment_height (int): Height of each image segment in pixels.
        @param region (float): The region to analyze for Y calibration.

        @return tuple: A tuple containing the status of calibration ("ok" or "error"),
                       the Y correction value in millimeters, and all the detected indexes.
        '''

        return self.__calibrate_y(images, mask, segment_height, region)

    def __calibrate_y(self, images, mask, segment_height, region):
        '''
        @brief Internal method to perform Y calibration for the given images.

        @param images (list): List of images to perform Y calibration on.
        @param mask (numpy.ndarray): The DFT mask used for image processing.
        @param segment_height (int): Height of each image segment in pixels.
        @param region (float): The region to analyze for Y calibration.

        @return tuple: A tuple containing the status of calibration ("ok" or "error"),
                       the Y correction value in millimeters, and all the detected indexes.
        '''

        # Convert region to segment units
        region = round(region / (2 * segment_height))

        # Evaluate image stack for horizontal segments
        h_segments_stack = super().eval_stack(images, mask, segment_height)

        # Calculate the number of segments
        nbr_segments = len(h_segments_stack[0])

        # List to store all detected indexes with maximum focus region
        all_indexes_max = []

        # List to store filtered indexes with maximum focus region
        filtered_indexes_max = []

        # Calculate the index with maximum focus region for each h_segment
        for h_segment in h_segments_stack:
            max_focus_region = 0
            idx_max_focus_region = 0
            for i in range(nbr_segments - region):
                focus_region = np.mean(h_segment[i:i + region])
                if focus_region > max_focus_region:
                    max_focus_region = focus_region
                    idx_max_focus_region = i + region // 2
            all_indexes_max.append(idx_max_focus_region)

        # Calculate the median of all detected indexes
        median_indexes = median(all_indexes_max)

        # Filter out indexes with low standard deviation
        for idx in all_indexes_max:
            if np.std([idx, median_indexes]) <= 2.5:
                filtered_indexes_max.append(idx)

        nbr_images = len(all_indexes_max)
        nbr_filtered_images = len(filtered_indexes_max)

        for idx in range(nbr_images):
          all_indexes_max[idx] = all_indexes_max[idx] * 2 * segment_height

        # Check if enough acceptable images are available for calibration
        if (nbr_images - nbr_filtered_images) > nbr_images / 2:
            return "error", f"too less of acceptable images available {nbr_filtered_images} / {nbr_images}", all_indexes_max

        # Calculate the Y correction value in millimeters
        y_correction_mm = (median(filtered_indexes_max) - nbr_segments // 2) * 2 * segment_height * 0.00274

        return "ok", y_correction_mm, all_indexes_max


class CalibrateZ(Calibrate):
    '''
    @class CalibrateZ
    Class to perform Z calibration.

    This class inherits from the `Calibrate` class and implements the Z calibration
    process for a given set of images.

    @attribute None

    @methods
    - __call__(self, images, mask, roi, roi_correction, segment_height):
        Perform Z calibration for the given images.

    - __calibrate_z(self, images, mask, roi, roi_correction, segment_height):
        Internal method to perform Z calibration for the given images.

    @usage
    ```
    index = CalibrateZ()(images, mask, roi, roi_correction, segment_height)
    ```
    '''

    def __call__(self, images, mask, roi, roi_correction, segment_height):
        '''
        @brief Perform Z calibration for the given images.

        @param images (list): List of images to perform Z calibration on.
        @param mask (numpy.ndarray): The DFT mask used for image processing.
        @param roi (tuple): Tuple representing the region of interest (ROI) as (p1, p2).
        @param roi_correction (int): Correction factor for the ROI in segments.
        @param segment_height (int): Height of each segment in pixels.

        @return index (int): Index of the calibrated Z position.
        '''
        return self.__calibrate_z(images, mask, roi, roi_correction, segment_height)

    def __calibrate_z(self, images, mask, roi, roi_correction, segment_height):
        '''
        @brief Internal method to perform Z calibration for the given images.

        @param images (list): List of images to perform Z calibration on.
        @param mask (numpy.ndarray): The DFT mask used for image processing.
        @param roi (tuple): Tuple representing the region of interest (ROI) as (p1, p2).
        @param roi_correction (int): Correction factor for the ROI in segments.
        @param segment_height (int): Height of each segment in pixels.

        @return index (int): Index of the calibrated Z position.
        '''

        h_segments_stack = super().eval_stack(images, mask, segment_height)

        (p1, p2) = roi
        nbr_segments = len(h_segments_stack[0])

        # Determine the start and end segments based on the ROI coordinates
        if p1 < p2:
            s_p1 = p1 // segment_height + roi_correction
            s_p2 = p2 // segment_height + roi_correction
        else:
            s_p1 = p2 // segment_height + roi_correction
            s_p2 = p1 // segment_height + roi_correction

        breite = (s_p2 - s_p1) // 2
        center = s_p1 + breite

        def __activation_seg(nbr_segment, center, breite, min_coeff, max_coeff):
            '''
            @brief Calculates the activation weights for segments within the ROI.

            This function calculates the activation weights for each segment within the ROI based on the given parameters.
            It applies a sigmoid-like activation function to assign weights to segments.

            @param nbr_segment (int): Total number of segments.
            @param center (int): Center segment index within the ROI.
            @param breite (int): Half-width of the ROI (in segments).
            @param min_coeff (float): Minimum weight value.
            @param max_coeff (float): Maximum weight value.

            @return weights (numpy.ndarray): Activation weights for each segment.
            '''

            s = (max_coeff - min_coeff) / breite

            def __sig(x):
                if x < center - breite:
                    return 0
                if x > center + breite:
                    return 0
                if x <= center:
                    return (x - (center - breite)) * s + min_coeff
                return max_coeff - (x - center) * s

            return np.array([__sig(x) for x in range(0, nbr_segment)], dtype=np.float32)

        dft_weights = __activation_seg(nbr_segments, center, breite, 0.8, 1)

        dft_classifier = []

        # Apply the activation weights to the focus evaluation results
        for idx_img in range(len(h_segments_stack)):
            dft_classifier.append(
                np.sum(np.multiply(h_segments_stack[idx_img], dft_weights))
            )

        dft_classifier = dft_classifier / max(dft_classifier)

        index = np.where(dft_classifier == 1)[0][0]

        return index



class Predict(Calibrate):
    '''
    @class Predict
    Class to perform prediction.

    This class inherits from the `Calibrate` class and implements the prediction
    process for a given set of images.

    @attribute None

    @methods
    - __call__(self, image, mask, segment_height, region, old_h_segment):
        Perform prediction for the given image.

    - __predict(self, image, mask, segment_height, region, old_h_segment):
        Internal method to perform prediction for the given image.

    @usage
    ```
    shift = Predict()(image, mask, segment_height, region, old_h_segment)
    ```
    '''

    def __call__(self, image, mask, segment_height, region, old_h_segment):
        '''
        @brief Perform prediction for the given image.

        @param image (numpy.ndarray): Image to perform prediction on.
        @param mask (numpy.ndarray): The DFT mask used for image processing.
        @param segment_height (int): Height of each segment in pixels.
        @param region (int): Number of segments to consider for prediction.
        @param old_h_segment (numpy.ndarray): Old focus evaluation results as 1D-array.

        @return shift (float): Predicted shift in Z position in mm.
        '''
        return self.__predict(image, mask, segment_height, region, old_h_segment)

    def __predict(self, image, mask, segment_height, region, old_h_segment):
        '''
        @brief Internal method to perform prediction for the given image.

        @param image (numpy.ndarray): Image to perform prediction on.
        @param mask (numpy.ndarray): The DFT mask used for image processing.
        @param segment_height (int): Height of each segment in pixels.
        @param region (int): Number of segments to consider for prediction.
        @param old_h_segment (numpy.ndarray): Old focus evaluation results as 1D-array.

        @return shift (float): Predicted shift in Z position in mm.
        '''

        FOCUS_in_mm = 0.02  # in mm :: 5% Focus-Differenz entspricht einen Verschiebung von 0.1 mm => 1% entspricht 0.02 mm

        new_h_segment = super().eval_stack(image, mask, segment_height)

        nbr_segments = len(old_h_segment)
        old_regions = []
        new_regions = []

        for idx in range(nbr_segments // 2 - region):
            old_regions.append(np.mean(old_h_segment[idx:idx + region + 1]))
            new_regions.append(np.mean(new_h_segment[idx:idx + region + 1]))

        index_best_old_region = old_regions.index(max(old_regions))
        index_best_new_region = new_regions.index(max(new_regions))

        focus_shift = []
        for idx in range(len(old_regions)):
            if index_best_new_region > index_best_old_region:
                focus_shift.append(old_regions[idx] / new_regions[idx])
            else:
                focus_shift.append(new_regions[idx] / old_regions[idx])

        shift = 100 * (1 - np.mean(focus_shift)) * FOCUS_in_mm

        if index_best_new_region > index_best_old_region:
            shift = (-1) * shift  # Move down

        return shift


