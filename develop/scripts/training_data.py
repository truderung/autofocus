import numpy as np
import cv2 as cv
import os

folder_path = '/Stack_1/Original_images'

saving_path = '/Stack_1/Transformed_images/trans_image_'

masks_path = '/Masks/mask'

# Masks Offsets and Coeffs:
offsets = [160, 330, 500, 660]
coeffs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]


def dft(img, offset, coeff):

    fenester_dft = cv.imread( masks_path + str(offset) + '_' + str(coeff) + '.png')

    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    f_shift = dft_shift * fenester_dft
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    img_back = np.sqrt(img_back)

    img_back = cv.normalize(
        img_back, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U
    )

    del dft
    del dft_shift
    del f_ishift
    del f_shift

    return img_back



# Bilde einlesen und in image_list speichen
image_list = []

for filename in os.listdir(folder_path):

    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)

        image = cv.imread(image_path)

        if image.shape[2] == 3 and len(image.shape) == 3:
            image = cv.cvtColor(
                image,
                cv.COLOR_BGR2GRAY,
            )
        elif len(image.shape) == 2:
            pass
        else:
            raise ValueError("Image format is not supported.")
        
        image_list.append(image)


# List of transformed images
trans_images_list = []


# The highest grayvalue in all transformed images
max_gray_value = 0


for image in image_list:

    trans_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    for o in offsets:
        for c in coeffs:

            inv_dft = dft(image, o, c)

            trans_image = trans_image + inv_dft

    max_trans_image = np.max(trans_image)

    max_gray_value = max_trans_image if max_gray_value < max_trans_image else max_gray_value

    trans_images_list.append(trans_image)



for index in range(len(trans_images_list)):

    trans_images_list[index] = trans_images_list[index] / max_gray_value

    cv.imwrite(
        saving_path +
        str(index) +
        '.png',
        trans_images_list[index],
    )
