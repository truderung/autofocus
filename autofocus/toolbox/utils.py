import cv2 as cv
import numpy as np
from numpy import linalg as la
from statistics import mean, median


PI_180 = np.pi / 180.0
_180_PI = 180 / np.pi


def create_DFT_mask(
    size,
    gamma_distribution=[45.0],
    level_distribution=[0.05, 0.11, 0.17, 0.22],
    width_distribution=[100],
):
    """
    @brief Creates a mask for Discrete Fourier Transform (DFT) based on specified parameters.

    This function generates a DFT mask with elliptical shapes based on the provided gamma, level, and width distributions.
    The generated mask is normalized to have values in the range [0, 1].

    @param size:               tuple       Size of the mask (height, width).
    @param gamma_distribution: list        List of gamma angles in degrees for elliptical shapes.
    @param level_distribution: list        List of level values as fractions of the diagonal for elliptical shapes.
    @param width_distribution: list        List of width values for elliptical shapes.

    @return: mask:             2D-array    The generated DFT mask.
    """

    mask_height, mask_width = size

    mask = np.zeros(size, dtype=np.uint16)
    diagonal = 2 * la.norm([mask_height, mask_width])

    def create_mask(gamma: float, level: int, width: int):
        """
        @brief Creates an elliptical mask based on gamma, level, and width.

        This function generates an elliptical mask with specified gamma, level, and width parameters.

        @param gamma:  float    Angle (in degrees) of the ellipse.
        @param level:  int      Level value as a fraction of the diagonal.
        @param width:  int      Width of the ellipse.

        @return: mask: 2D-array The generated elliptical mask.
        """

        # we round the angle here because cv.ellipse rounds inside the function
        gamma_bar = round(
            np.arctan(np.tan(gamma * PI_180) * mask_width / mask_height) * _180_PI
        )
        x_shift = round(np.sin(gamma_bar * PI_180) * diagonal)
        y_shift = round(np.cos(gamma_bar * PI_180) * diagonal)

        mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
        cv.ellipse(
            mask,
            (x_shift, y_shift),
            (width, round(diagonal - 2 * level)),
            -gamma_bar,
            180,
            360,
            255,
            -1,
        )

        return mask

    for g in gamma_distribution:
        for l in list(np.array(level_distribution) * diagonal):
            l = round(l)
            for w in width_distribution:
                mask += create_mask(g, l, w)

    mask = mask / np.max(mask)

    return mask



def heat(dft_img, mask):
    """
    @brief Heats the image using Discrete Fourier Transform (DFT).

    This function applies the provided mask to the DFT image, performs inverse DFT, and calculates the magnitude.

    @param dft_img: 3D-array      DFT of the original image.
    @param mask:    2D-array      Mask to be applied during the heating process.

    @return: res: 2D-array      Resulting heated image.
    """

    res = dft_img * np.dstack((mask, mask))
    res = cv.idft(res)
    res = cv.magnitude(res[:, :, 0], res[:, :, 1])

    return res



def heat_double(orig_img, mask):
    """
    @brief Heats the image in two halves using Discrete Fourier Transform (DFT).

    This function divides the original image into two halves and applies the heating process to each half separately.
    It computes the DFT of the image, restricts it to the left and right halves, and applies the corresponding masks.
    The resulting heated images for each half are returned.

    @param orig_img: 2D-array      Original image.
    @param mask:     2D-array      Mask to be applied during the heating process.

    @return: res:      2D-array   Resulting heated image for the left half.
    @return: res_bar:  2D-array   Resulting heated image for the right half.
    """

    h, w = orig_img.shape[0] // 2, orig_img.shape[1] // 2
    img = cv.dft(np.float64(orig_img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_img = img[0:h, 0:w, :]
    dft_img_bar = img[0:h, w:, :]
    mask_bar = cv.flip(mask, flipCode=1)

    res = heat(dft_img, mask)
    res_bar = heat(dft_img_bar, mask_bar)

    return res, res_bar



def horz_eval(img, weights, seg_height: int = 10):
    """
    @brief Evaluates the focus distribution along the vertical axis in an image.

    The image is divided into multiple horizontal segments, where each segment has the same width as the image and a height defined by the `seg_height` parameter.

    @param img:        2D-array      Input image. Should be a one-channel grayscale image.
    @param weights:    1D-array      Weight values used for focus evaluation. Determines the contribution of different gray values.
    @param seg_height: int           Defines the segment height. Defaults to 40.

    @throws ValueError: If the input image has more than one channel.

    @return seg_focus: 1D-array      Contains the focus evaluation for each segment.
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



def activation(hist_stack, otsu_threshold):
    """
    @brief Calculates the activation function for histogram-based image thresholding.

    The activation function is computed based on the input histogram stack and the Otsu threshold value (if provided).
    If the Otsu threshold is not provided, the activation function is computed using the mean and standard deviation of the histogram stack.

    @param hist_stack:      2D-array      Histogram stack containing cumulative histograms for multiple images.
    @param otsu_threshold:  int or None   Otsu threshold value. If provided, it will be used as the activation threshold.

    @return filt_1:         1D-array      Activation function values for each gray level.
    """
    if otsu_threshold:
        threshold = otsu_threshold
    else:
        hists = hist_stack.cumsum(1)[:, -1]
        hist_mean = np.mean(hists)
        hist_std = np.std(hists)

        filt = hists[hists < hist_mean + hist_std]
        threshold = len(hists) - len(filt)

    s = 1 / (255 - threshold)

    def __Sigmoid(x: int):
        """
        @brief Sigmoid function used for calculating the activation value.

        This function calculates the activation value based on the input x and threshold.

        @param x:         int       Input value (gray level).
        @param threshold: int       Activation threshold.

        @return:          float     Activation value.
        """
        if x < threshold:
            return 0
        return (x - threshold) * s

    weights = np.array([__Sigmoid(x) for x in range(0, 256)], dtype=np.float64)

    return weights



def eval_stack(heat_maps, max_value, heat_maps_bar, max_value_bar, segment_height):
    """
    @brief Evaluates the stack of heat maps and computes focus evaluation for each segment.

    This function takes two stacks of heat maps, their corresponding maximum values, and the segment height as input.

    It calculates the histograms for each heat map, computes the activation weights,
        and evaluates the focus distribution for each segment using the `horz_eval` function.

    @param heat_maps:       List of 2D-arrays   Stack of heat maps.
    @param max_value:       float               Maximum value of the heat maps.
    @param heat_maps_bar:   List of 2D-arrays   Stack of heat maps (bar).
    @param max_value_bar:   float               Maximum value of the heat maps (bar).
    @param segment_height:  int                 Height of each segment.

    @return h_s_stack:      List of 1D-arrays   Focus evaluation for each segment.
    """

    def __run_work(heat_maps, max_value):
        """
        @brief Helper function to compute focus evaluation for a stack of heat maps.

        This function calculates the histograms for each heat map, computes the activation weights using the `activation` function,
        and evaluates the focus distribution for each segment using the `horz_eval` function.

        @param heat_maps:           List of 2D-arrays   Stack of heat maps.
        @param max_value:           float               Maximum value of the heat maps.

        @return max_focus:          float               Maximum focus value.
        @return h_segments_stack:   List of 1D-arrays   Focus evaluation for each segment.
        """
        hist_stack = np.empty(shape=(256, 0), dtype=np.uint32)
        heat_map_stack = []
        # print("Load Heatmaps for normalization and histogramm calculation............")
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

        # print("Normalization and histogramm calculation Done............")
        # print("Weights calculation............")
        weights = activation(hist_stack, otsu_threshold=False)
        # print("Weights calculation Done............")
        max_focus = 0
        h_segments_stack = []
        # print("Start horiz evaluation............")
        for heat_map in heat_map_stack:
            seg_focus = horz_eval(heat_map, weights, segment_height)
            max_focus = np.max([max_focus, np.max(seg_focus)])
            h_segments_stack += [seg_focus]
        # print("Start horiz evaluation Done............")
        return max_focus, h_segments_stack

    h_segments_stack_double = [
        __run_work(heat_maps, max_value),
        __run_work(heat_maps_bar, max_value_bar),
    ]

    # print("Unite Left and Right............")
    h_segments_stack_united = []
    for idx, seg_focus in enumerate(h_segments_stack_double[0][1]):
        seg_focus_bar = h_segments_stack_double[1][1][idx]
        seg_focus = seg_focus / h_segments_stack_double[0][0]
        seg_focus_bar = seg_focus_bar / h_segments_stack_double[1][0]
        h_segments_stack_united += [np.amax([seg_focus, seg_focus_bar], axis=0)]
    # print("Unite Left and Right Done............")
    return h_segments_stack_united


def evaluate_roi(roi, roi_correction, h_segments_stack, segment_height):
    """
    @brief Evaluates the segments within the region of interest (ROI) based on the given focus evaluation results.

    This function takes the ROI, focus evaluation results (h_segments_stack), and segment height as input.
    It calculates the activation weights for the segments within the ROI and applies them to the focus evaluation results.
    The function returns the indices of segments that have a classification value of 1.

    @param roi:              tuple       Region of interest (p1, p2).
    @param h_segments_stack: List of 1D-arrays  Focus evaluation results for each segment.
    @param segment_height:   int         Height of each segment.

    @return: indices:        1D-array    Indices of segments with a classification value of 1.
    """

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

    def activation_seg(nbr_segment, center, breite, min_coeff, max_coeff):
        """
        @brief Calculates the activation weights for segments within the ROI.

        This function calculates the activation weights for each segment within the ROI based on the given parameters.
        It applies a sigmoid-like activation function to assign weights to segments.

        @param nbr_segment: int       Total number of segments.
        @param center:      int       Center segment index within the ROI.
        @param breite:      int       Half-width of the ROI (in segments).
        @param min_coeff:   float     Minimum weight value.
        @param max_coeff:   float     Maximum weight value.

        @return: weights:   1D-array  Activation weights for each segment.
        """
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

    dft_weights = activation_seg(nbr_segments, center, breite, 1, 0.8)

    dft_classifier = []

    # Apply the activation weights to the focus evaluation results
    for idx_img in range(len(h_segments_stack)):
        dft_classifier.append(
            np.sum(np.multiply(h_segments_stack[idx_img], dft_weights))
        )

    dft_classifier = dft_classifier / max(dft_classifier)

    indices = np.where(dft_classifier == 1)[0][0]

    return indices


def calibrate_y(h_segments_stack):
    """
    @brief Calibrates the Y-axis based on the focus evaluation results.

    This function calibrates the Y-axis based on the focus evaluation results for each segment.
    It identifies the peak segment that represents the sharpest focus and returns the calibration result.

    @param h_segments_stack: List of 1D-arrays  Focus evaluation results for each segment.
    @param segment_height:   int             Height of each segment.

    @return: calibration_result:    bool      True if calibration is successful, False if calibration is needed.
    @return: y_correction_mm:       float     Y-axis correction in millimeters.
    @return: y_correction:          int       Y-axis correction in segment units.
    """

    nbr_segments = len(h_segments_stack[0])
    # print(f"Number of segments: {nbr_segments}")

    AUSNAHME = True
    FIFTH_ITER = True
    SAME_POSITION = True
    PEAK = False
    idx = 0
    ignored_images = 0
    indexes = []

    # print("Start Y_Calibration............")

    while SAME_POSITION and not PEAK:
        if idx - ignored_images >= 5:
            FIFTH_ITER = False

        max_value = np.max(h_segments_stack[idx])
        idx += 1
        # print(f"Iteration {idx-1}: Max value={max_value}")
        if max_value > 0.75:
            PEAK = True
        elif max_value < 0.2:
            ignored_images += 1
        else:
            idx_max = h_segments_stack[idx].argmax()
            # print(f"Iteration {idx-1}: Index={idx_max}")
            if (
                FIFTH_ITER
                and (abs(idx_max - nbr_segments // 2) > 10)
                and (max_value > 0.5)
            ):
                indexes = []
                break
            else:
                indexes.append(idx_max)
                if np.std(indexes) > 2:
                    if AUSNAHME:
                        indexes.pop(-1)
                        AUSNAHME = False
                    else:
                        # print(f"Iteration {idx-1}: BREAK because STD={np.std(indexes)}")
                        indexes.pop(-1)
                        SAME_POSITION = False

    if len(indexes) < 5:
        nbr_new_images = 5 - len(indexes)
        return False, None, nbr_new_images
    else:
        y_correction = int(median(indexes) - nbr_segments // 2)
        return True, y_correction, len(indexes)


def calibrate_y_2(h_segments_stack, region):

    # print("##################################################################")
    # print("Start Y_Calibration_2............")

    nbr_segments = len(h_segments_stack[0])
    all_indexes_max = []
    indexes_max = []

    for h_segment in h_segments_stack:
        max_focus_region = 0
        idx_max_focus_region = 0
        for i in range(nbr_segments-region):
            focus_region = np.mean(h_segment[i:i+region])
            if focus_region > max_focus_region:
                max_focus_region = focus_region
                idx_max_focus_region = i + region//2
        all_indexes_max.append(idx_max_focus_region)
   
    median_indexes = median(all_indexes_max)
    # std_indexes = np.std(all_indexes_max)

    # print("All indexes: ", all_indexes_max)
    # print(f"median_indexes : {median_indexes}")
    # print(f"std_indexes : {std_indexes}")

    for idx in all_indexes_max:
        if np.std([idx, median_indexes]) <= 2:
            indexes_max.append(idx)
    # print("Filtered indexes: ", indexes_max)
    return (median(indexes_max) - nbr_segments//2) * 20


def predict(old_h_segment, new_h_segment, region): # (old_h_segment, region): #

    FOCUS_in_mm = 0.02 # in mm :: 5% Focus-Differenz entspricht einen Verschiebung von 0.1 mm => 1% entspricht 0.02 mm

    nbr_segments = len(old_h_segment)
    old_regions = []
    new_regions = []

    for idx in range(nbr_segments//2 - region):
        old_regions.append(mean(old_h_segment[idx:idx+region+1]))
        new_regions.append(mean(new_h_segment[idx:idx+region+1]))
    
    index_best_old_region = old_regions.index(max(old_regions))
    index_best_new_region = new_regions.index(max(new_regions))

    focus_shift = []
    for idx in range(len(old_regions)):
        if index_best_new_region > index_best_old_region:
            focus_shift.append(old_regions[idx] / new_regions[idx])
        else:
            focus_shift.append(new_regions[idx] / old_regions[idx])

    shift = 100 * (1 - mean(focus_shift)) * FOCUS_in_mm
    
    if index_best_new_region > index_best_old_region:
        shift = (-1) * shift    #Move down

    return shift

"""
@brief Calibrates the Y-axis based on the focus evaluation results.

This function calibrates the Y-axis based on the focus evaluation results for each segment.
It identifies the peak segment that represents the sharpest focus and returns its index for calibration.

@param h_segments_stack: List of 1D-arrays  Focus evaluation results for each segment.

@return: peak:           int or None        Index of the peak segment for calibration. None if calibration was not performed.
"""


'''  

class Heater:
    """
    @brief Class for heating an image using Discrete Fourier Transform (DFT).

    The Heater class takes an original image, a half-size, and a mask as input. It performs operations using the DFT to heat the image.

    The heating process is divided into two parts: `heat()` and `heat_bar()`. Each part operates on different regions of the DFT image.

    Usage:
    - Create an instance of the Heater class with the required parameters.
    - Call the `heat()` method to heat the left top quarter of the image using the provided mask.
    - Call the `heat_bar()` method to heat the right top quarter of the image using the flipped mask.

    Note: This class assumes the availability of the cv and np modules.

    @param orig_img:    2D-array      Original image.
    @param half_size:   tuple         Half size (height, width) of the region to operate on.
    @param mask:        2D-array      Mask to be applied during the heating process.

    @var dft_img:       3D-array      DFT of the original image, restricted to the left top quarter.
    @var dft_img_bar:   3D-array      DFT of the original image, restricted to the right top quarter.
    @var mask_bar:      2D-array      Flipped mask used in the `heat_bar()` method.

    @return: res:       2D-array      Resulting heated image.
    """

    def __init__(self, orig_img, half_size, mask):
        """
        @brief Initializes the Heater object.

        Initializes the Heater object by performing the DFT on the original image, restricting it to the left top quarter and right top quarter.
            Also stores the provided mask and its flipped version.

        @param orig_img:    2D-array      Original image.
        @param half_size:   tuple         Half size (height, width) of the region to operate on.
        @param mask:        2D-array      Mask to be applied during the heating process.
        """

        h, w = self.size = half_size
        # create dft from image, but operate only on the left top, quarter of the image
        img = cv.dft(np.float64(orig_img), flags=cv.DFT_COMPLEX_OUTPUT)
        self.dft_img = img[0:h, 0:w, :]
        self.dft_img_bar = img[0:h, w:, :]
        self.mask = mask
        self.mask_bar = cv.flip(mask, flipCode=1)


    def heat(self):
        """
        @brief Heats the left top quarter of the image.

        Heats the left top quarter of the image by multiplying the DFT image (restricted to the left top quarter) with the mask, 
            applying inverse DFT, and calculating the magnitude.

        @return: res:   2D-array      Resulting heated image.
        """       
        res = self.dft_img * np.dstack((self.mask, self.mask))
        res = cv.idft(res)
        res = cv.magnitude(res[:, :, 0], res[:, :, 1])
        return res


    def heat_bar(self):
        """
        @brief Heats the right top quarter of the image.

        Heats the right top quarter of the image by multiplying the DFT image (restricted to the right top quarter) with the flipped mask, 
            applying inverse DFT, and calculating the magnitude.

        @return: res:   2D-array      Resulting heated image.
        """
        res = self.dft_img_bar * np.dstack((self.mask_bar, self.mask_bar))
        res = cv.idft(res)
        res = cv.magnitude(res[:, :, 0], res[:, :, 1])
        return res

    ......................................


    nbr_images = len(h_segments_stack)

    for_calibration = []
    not_for_calibration = []

    # Find the maximum focus value and its corresponding segment index for each image
    for i in range(nbr_images):
        max_value = max(h_segments_stack[i])
        max_idx = h_segments_stack[i].index(max_value)

        if max_value > 0.95:
            not_for_calibration += i
        else:
            for_calibration += max_idx
    
    if len(not_for_calibration) != 0:
        new_pos = not_for_calibration[0] + 0.1*nbr_images
        nbr_new_images = nbr_images - not_for_calibration[0]
        print(f"Move to {new_pos} and take {nbr_new_images} images")

        return None
    
    else:
        peak = int(median(for_calibration))

        return peak
    '''
