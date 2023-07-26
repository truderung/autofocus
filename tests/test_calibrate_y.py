from autofocus.toolbox.calibrators import *
from autofocus.toolbox.masker import Masker
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


  Path(IMAGE_PATH, 'Stack_5/Original_images/25.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/24.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/23.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/22.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/21.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/20.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/19.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/18.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/17.jpg'),
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

IMG_STACK_Y_CALIB = [
  Path(IMAGE_PATH, 'Stack_5/Original_images/51.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/50.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/49.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/48.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/47.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/46.jpg'),  
  Path(IMAGE_PATH, 'Stack_5/Original_images/45.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/44.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/43.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/42.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/41.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/40.jpg'),
  Path(IMAGE_PATH, 'Stack_5/Original_images/39.jpg')
]


IMG_STACK_Z_CALIB = [
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
  Path(IMAGE_PATH, 'Stack_5/Original_images/26.jpg')
]


SEG_HEIGHT = 10

print("############################################")
print("##   Init ")

g = 46
d = 7
gamma_range = list(range(g-d, g+d+1))
level_distribution = [0.005, 0.01, 0.02, 0.05, 0.1]
width_distribution = [30, 60, 100]


masker = Masker((1400, 2660))
mask = masker.get_mask(gamma_range, level_distribution, width_distribution)


y_images = []

for p in IMG_STACK_Y_CALIB:
    img = cv.imread(p.as_posix(), cv.IMREAD_GRAYSCALE)
    y_images.append(img)

z_images = []

for p in IMG_STACK_Y_CALIB:
    img = cv.imread(p.as_posix(), cv.IMREAD_GRAYSCALE)
    z_images.append(img)


print("############################################")
print("##   Y_Calibration ")

y_flag, y_correction, all_indexes = CalibrateY()(y_images, mask, SEG_HEIGHT, 770)

if y_flag == "error": print(y_correction)
if y_flag == "ok": print(f"y_correction = {y_correction}")

print(f"All indexes: {all_indexes}")


print("############################################")
print("##   Z_Calibration ")

roi = (700//2, 2100//2)

roi_correction = round(y_correction/(2 * SEG_HEIGHT * 0.00274))

index, old_h_segment = CalibrateZ()(z_images, mask, roi, roi_correction, SEG_HEIGHT)

print(f"Z_Correction = {index}")


print("############################################")
print("##   Predict ")

p = Path(IMAGE_PATH, 'Stack_5/Original_images/31.jpg')
image = cv.imread(p.as_posix(), cv.IMREAD_GRAYSCALE)


shift = Predict()(image, mask, SEG_HEIGHT, 770, old_h_segment)

print(f"Shift = {shift}")

'''
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

    print("##################################################################")
    print("Start Y_Calibration............")

    while SAME_POSITION and not PEAK:
        if idx - ignored_images >= 5:
            FIFTH_ITER = False

        max_value = np.max(h_segments_stack[idx])
        idx += 1
        print(f"Iteration {idx-1}: Max value={max_value}")
        if max_value > 0.75:
            PEAK = True
        elif max_value < 0.2:
            ignored_images += 1
        else:
            idx_max = h_segments_stack[idx].argmax()
            print(f"Iteration {idx-1}: Index={idx_max}")
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
                        print(f"Iteration {idx-1}: BREAK because STD={np.std(indexes)}")
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

    return mask'''



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
