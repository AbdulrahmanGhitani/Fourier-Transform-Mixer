import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class ImageViewer(QGraphicsView):
    def __init__(self, path):

        self._original_image = cv2.imread(f'{path}')
        self._gray_scale_image = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2GRAY)
        self._image_size = None
        self._image_brightness = 1 # The common range for alpha is from 0.0 to 2.0. (1 ==> no change)
        self._image_contrast = 0 # The common range for beta is from  -255 to 255. (0 ==> no change)
        self._image_fft = np.fft.fft2(self._gray_scale_image)
        self._image_fft_shift = np.fft.fftshift(self.image_fft)
        self._image_ifft = None
        self._original_image_magnitude = None
        self._weight_of_magnitude = 1
        self._equalized_image_magnitude = self.original_image_magnitude
        self._original_image_phase = None
        self._weight_of_phase = 1
        self._equalized_image_phase = self.original_image_phase
        self._image_real_part = None
        self._image_imaginary_part = None

    def convert_to_grayscale(self, pixmap):
        image = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        grayscale_pixmap = QPixmap.fromImage(image)
        return grayscale_pixmap
    @property
    def original_image(self):
        return self._original_image

    @property
    def gray_scale_image(self):
        self._gray_scale_image = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2GRAY)
        return self._gray_scale_image

    @property
    def image_size(self):
        self._image_size = self._original_image.shape
        return self._image_size[:2]

    @image_size.setter
    def image_size(self, value):
        self._original_image = cv2.resize(self._original_image, value)
        self._image_size = value

    @property
    def image_contrast(self):
        return self._image_contrast

    @image_contrast.setter
    def image_contrast(self, value):
        if -255 <= value <= 255:
            self._image_contrast = value
            self._gray_scale_image = cv2.convertScaleAbs(self._original_image, alpha=self.image_brightness,
                                                              beta=self.image_contrast)

    @property
    def image_brightness(self):
        return self._image_brightness

    @image_brightness.setter
    def image_brightness(self, value):
        if 0 <= value <= 2:
            self._image_brightness = value
            self._gray_scale_image = cv2.convertScaleAbs(self._original_image, alpha=self.image_brightness,
                                                                  beta=self.image_contrast)

    @property
    def image_fft(self):
        return self._image_fft

    @property
    def image_fft_shift(self):
        return self._image_fft_shift

    @image_fft_shift.setter
    def image_fft_shift(self, value):
        pass

    @property
    def original_image_magnitude(self):
        self._original_image_magnitude = 20 * np.log(np.abs(self.image_fft_shift))
        return self._original_image_magnitude

    @property
    def equalized_image_magnitude(self):
        self._equalized_image_magnitude *= self._weight_of_magnitude
        return self._equalized_image_magnitude

    @property
    def weight_of_magnitude(self):
        return self._weight_of_magnitude

    @weight_of_magnitude.setter
    def weight_of_magnitude(self, value):
        self._weight_of_magnitude = value / 10
        # self._equalized_image_magnitude = self.equalized_image_magnitude

    @property
    def original_image_phase(self):
        self._original_image_phase = np.angle(self._image_fft_shift)
        return self._original_image_phase

    @property
    def equalized_image_phase(self):
        self._equalized_image_phase *= self._weight_of_phase
        return self.equalized_image_phase

    @property
    def weight_of_phase(self):
        return self._weight_of_phase

    @weight_of_phase.setter
    def weight_of_phase(self, value):
        self._weight_of_phase = value / 10
        # self._equalized_image_phase = self.equalized_image_phase

    @property
    def image_real_part(self):
        self._image_real_part = 20 * np.log(np.real(self._image_fft_shift))
        return self._image_real_part

    @property
    def image_imaginary_part(self):
        self._image_imaginary_part = np.imag(self._image_fft_shift)
        return self._image_imaginary_part



class MixImage(object):
    def __init__(self, image1, image2, image3, image4):
        self._input_images = [image1, image2, image3, image4]
        self._mixed_fft = np.zeros_like(np.fft.fft2(self._input_images[0]), dtype=np.complex128)
        self._mixed_image = None

    # def customize_images_size(self):
    #     small_size_image = (100000, 100000)
    #     for i in range(4):
    #         if self._input_images[i].image_size < small_size_image
    #             small_size_image = self._input_images[i].image_size
    #     print(small_size_image)
    def mix_images(self):
        for i in range(4):
            weighted_fft = (
                    self._input_images[i].weight_of_magnitude * np.abs(self._input_images[i].image_fft)
                    * np.exp(1j * self._input_images[i].weight_of_phase)
            )
            self._mixed_fft += weighted_fft
        self._mixed_image = np.abs(np.fft.ifft2(self._mixed_fft)).astype(np.uint8)
        return self._mixed_image


# image_1 = ImageViewer('images.jpeg')
# image_2 = ImageViewer('download.jpeg')
# image_3 = ImageViewer('images (1).jpeg')
# image_4 = ImageViewer('download (1).jpeg')
#
# image_1.image_size = (500,500)
# image_2.image_size = (500,500)
# image_3.image_size = (500,500)
# image_4.image_size = (500,500)
#
# # image_1.weight_of_magnitude = 10
# # image_2.weight_of_magnitude = 10
# # image_3.weight_of_magnitude = 10
# # image_4.weight_of_magnitude = 10
# #
# # image_1.weight_of_phase = 10
# # image_2.weight_of_phase = 10
# # image_3.weight_of_phase = 10
# # image_4.weight_of_phase = 10
#
# mixed_image = MixImage(image_1, image_2, image_3, image_4)
# # cv2.imread('images.jpeg', mixed_image.mix_images())
#
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
