import cv2
import numpy as np
from PyQt5.QtGui import QImage

class ImageViewer(object):
    def __init__(self, path):
        self._original_image = cv2.imread(f'{path}')
        self.gray_scale_image = cv2.cvtColor(self._original_image, cv2.COLOR_BGR2GRAY)
        self.image_size = None
        self.image_brightness = None
        self.image_contrast = None
        self._image_fft_shift = None
        self._image_ifft = None
        self._original_image_magnitude = None
        self.equalized_image_magnitude = None
        self._original_image_phase = None
        self.equalized_image_phase = None
        self._image_real_part = None
        self._image_imaginary_part = None

    @property
    def original_image(self):
        return self._original_image

    @property
    def gray_scale_image(self):
        return self._gray_scale_image

    @property
    def image_size(self):
        return self._original_image.shape

    @image_size.setter
    def image_size(self, width, height):
        self._original_image = cv2.resize(self._original_image, (width, height))

    @property
    def image_contrast(self):
        return self.image_contrast

    @image_contrast.setter
    def image_contrast(self, value):
        self._original_image = cv2.convertScaleAbs(self._original_image, beta=value)

    @property
    def image_brightness(self):
        return self.image_brightness

    @image_brightness.setter
    def image_brightness(self, value):
        self._original_image = cv2.convertScaleAbs(self._original_image, alpha=value)

    @property
    def image_fft_shift(self):
        fft_2D = np.fft.fft2(self._original_image)
        self._image_fft_shift = np.fft.fftshift(fft_2D)
        return self._image_fft_shift

    @image_fft_shift.setter
    def image_fft_shift(self, value):
        pass

    @property
    def original_image_magnitude(self):
        self._original_image_magnitude = 20 * np.log(np.abs(self._image_fft_shift))
        return self._original_image_magnitude

    @property
    def equalized_image_magnitude(self):
        return self.equalized_image_magnitude

    @equalized_image_magnitude.setter
    def equalized_image_magnitude(self, value):
        self.equalized_image_phase = self.equalized_image_phase * (value/10)

    @property
    def original_image_phase(self):
        self._original_image_phase = np.angle(self._image_fft_shift)
        return self._original_image_phase

    @property
    def image_real_part(self):
        self._image_real_part = 20 * np.log(np.real(self._image_fft_shift))
        return self._image_real_part

    @property
    def image_imaginary_part(self):
        self._image_imaginary_part = np.imag(self._image_fft_shift)
        return self._image_imaginary_part
