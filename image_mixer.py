import cv2
import numpy as np
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
from PyQt5.uic import loadUiType
import cv2
import matplotlib.pyplot as plt


class ViewOriginal(QGraphicsView):
    imageSelected = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = None
        self.original_pixmap = None
        self.grayscale_pixmap = None
        self.path = self.path()
        self.image_viewer = None # ImageViewer object

    # def resize_pixmap(self, new_width, new_height):
    #     if self.original_pixmap:
    #         self.original_pixmap = self.original_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)
    #         self.grayscale_pixmap = self.convert_to_grayscale(self.original_pixmap)

    # def convert_to_grayscale(self, pixmap):
    #     image = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
    #     grayscale_pixmap = QPixmap.fromImage(image)
    #     return grayscale_pixmap

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.openImageDialog()

    def mouseMoveEvent(self, event: QMouseEvent):
        # You can capture the mouse position during the drag here
        brightness = abs(self.mapToScene(event.pos()).x()/1000)*2
        contrast = (self.mapToScene(event.pos()).y()/500)*255
        if brightness <=1 and brightness >=0 :
            self.image_viewer.image_brightness = brightness
        if contrast <= 1 and contrast >=0:
            self.image_viewer.image_contrast = contrast
        image_bytes = self.image_viewer.gray_scale_image_bytes()
        
        self.original_pixmap.loadFromData(image_bytes.tobytes())
        # self.resize_pixmap(300, 200)

        if not self.pixmap_item:
            self.pixmap_item = QGraphicsPixmapItem(self.original_pixmap)
            self.scene.addItem(self.pixmap_item)

        else:
            self.pixmap_item.setPixmap(self.original_pixmap)

        print("Mouse position during drag:", self.mapToScene(event.pos()))

    def openImageDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Image Files (*.png *.jpg *.bmp *.jpeg *.tif *.tiff);;All Files (*)",
                                                  options=options)
        if fileName:
            self.loadImage(fileName)
            self.imageSelected.emit(fileName)  # Emit the filename
            print(fileName)

            return fileName

    def path(self):
        path=self.openImageDialog
        # print(path)
        return path

    def loadImage(self, filename):
        #self.original_pixmap = QPixmap(filename)
        self.original_pixmap = QPixmap()
        self.image_viewer = ImageViewer(filename)
        image_bytes = self.image_viewer.gray_scale_image_bytes()
        
        self.original_pixmap.loadFromData(image_bytes.tobytes())
        # self.resize_pixmap(300, 200)

        if not self.pixmap_item:
            self.pixmap_item = QGraphicsPixmapItem(self.original_pixmap)
            self.scene.addItem(self.pixmap_item)

        else:
            self.pixmap_item.setPixmap(self.original_pixmap)

class ViewWeight(QGraphicsView):
    def __init__(self):
        super().__init__()

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image = None # image_viewer 
        self.qt_image = None
        self.current_pixmap = None
        # self.grayscale_pixmap = None
        self.resized_photo=None
        self.current_pixmap_item =None
        self._current_state = 'm'
        self._current_image = None
        self._current_clipped_image = None
       

        self.drawing_rectangle = False
        self.rectangle_item = None
        self.start_point = None
        self._rect_x_limits = None
        self._rect_y_limits = None

    @property
    def rect_x_limits(self):
        return self._rect_x_limits
    @rect_x_limits.setter
    def rect_x_limits(self,value):
        self._rect_x_limits = value
    @property
    def rect_y_limits(self):
        return self._rect_y_limits
    @rect_y_limits.setter
    def rect_y_limits(self,value):
        self._rect_y_limits = value

    @property
    def current_state(self):
        return self._current_state
    
    @current_state.setter
    def current_state(self, value):
        self._current_state = value
        if value == 'm':
            self.current_image = self.image.original_image_magnitude
        elif value == 'p':
            self.current_image = self.image.original_image_phase
        elif value == 'r':
            self.current_image = self.image.image_real_part
        elif value == 'i':
            self.current_image = self.image.image_imaginary_part
        else:
            raise ValueError('Invalid state')
        self.qt_image = self.convert_cv_to_qt(self.current_image.astype(np.uint8))
        self.current_pixmap = QPixmap(self.qt_image)
        # self.grayscale_pixmap = self.convert_to_grayscale(self.original_pixmap)
        # self.resized_photo=self.resize_pixmap(300,200)
        # self.resize_pixmap(300,200)
        # self.pixmap_item = QGraphicsPixmapItem(self.grayscale_pixmap)
        if self.current_pixmap_item:
            self.current_pixmap_item.setPixmap(self.current_pixmap)
        else:
            self.current_pixmap_item = QGraphicsPixmapItem(self.current_pixmap)
            self.scene.addItem(self.current_pixmap_item)
        
    @property
    def current_image(self):
        return self._current_image
    
    @current_image.setter
    def current_image(self, value):
        self._current_image = value
    @property
    def current_clipped_image(self):
        clipped_image = np.zeros_like(self.current_image)
        x_i = self.rect_x_limits[0]
        x_f = self.rect_x_limits[1]
        y_i = self.rect_y_limits[0]
        y_f = self.rect_y_limits[1]
        clipped_image[y_i:y_f,x_i:x_f] = self.current_image[y_i:y_f,x_i:x_f]
        self._current_clipped_image = clipped_image
        return self._current_clipped_image
    

    # def convert_to_grayscale(self, pixmap):
    #     image = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
    #     grayscale_pixmap = QPixmap.fromImage(image)
    #     return grayscale_pixmap

    # def resize_pixmap(self, new_width, new_height):
    #     self.current_pixmap = self.current_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)
        # self.grayscale_pixmap = self.convert_to_grayscale(self.original_pixmap)

    def convert_cv_to_qt(self, cv_image):
        height, width = cv_image.shape
        bytes_per_line = width
        qt_image = QImage(cv_image.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qt_image)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = self.mapToScene(event.pos())
            self.drawing_rectangle = True

    def mouseMoveEvent(self, event):
        if event.button() != Qt.LeftButton:
            if self.drawing_rectangle:
                current_point = self.mapToScene(event.pos())
                rect = QRectF(self.start_point, current_point).normalized()
                self.rect_x_limits = (int(self.start_point.x()),int(current_point.x()))
                self.rect_y_limits = (int(self.start_point.y()),int(current_point.y()))
              
                if not self.rectangle_item:
                    self.rectangle_item = QGraphicsRectItem(rect)
                    pen = QPen(QColor(Qt.white), 2,
                               Qt.DashLine)  # You can use Qt.DashLine or Qt.DotLine for a dashed or dotted line
                    self.rectangle_item.setPen(pen)
                    self.scene.addItem(self.rectangle_item)
                else:
                    self.rectangle_item.setRect(rect)
        else:
            print("Mouse position during drag:",self.mapToScene(event.pos()))

    def mouseReleaseEvent(self, event):
        if self.drawing_rectangle:
            self.drawing_rectangle = False
            self.start_point = None
            # np.savetxt('array2.csv',self.current_clipped_image,delimiter=',',fmt="%d")
            # np.savetxt('array1.csv',self.current_image,delimiter=',',fmt="%d")
            # Get information about the points inside the rectangle
            # points_in_rectangle = self.get_points_in_rectangle(self.rectangle_item.rect())
            # print("Points inside the rectangle:", points_in_rectangle)

    # def get_points_in_rectangle(self, rectangle):
    #     # Get the region of interest from the image
    #     region_of_interest = self.grayscale_pixmap.toImage().copy(rectangle.toRect()).convertToFormat(QImage.Format_Grayscale8)
    #     # Get the pixel values within the rectangle
    #     points = []
    #     for y in range(region_of_interest.height()):
    #         for x in range(region_of_interest.width()):
    #             pixel_value = region_of_interest.pixel(x, y)
    #             points.append((x + int(rectangle.x()), y + int(rectangle.y()), QColor(pixel_value).getRgb()))
    #     #print(pixel_value)
    #     return points
    

class ImageViewer():
    def __init__(self, path):

        self._original_image = cv2.resize(cv2.imread(f'{path}'),(300,200),cv2.INTER_LINEAR)
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
        

    @property
    def original_image(self):
        return self._original_image

    @property
    def gray_scale_image(self):
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
        self._original_image_magnitude = 20 * np.log10(np.abs(self.image_fft_shift))
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
        self._image_real_part = 20 * np.log10(np.real(self._image_fft_shift))
        return self._image_real_part

    @property
    def image_imaginary_part(self):
        self._image_imaginary_part = np.imag(self._image_fft_shift)
        return self._image_imaginary_part
    def gray_scale_image_bytes(self):
        _,data = cv2.imencode(img=self.gray_scale_image, ext='.jpeg')
        return data
    

class MixImage(object):
    def __init__(self, image1, image2, image3, image4):
        self._input_images = [image1, image2, image3, image4]
        self._mixed_fft = np.zeros_like(np.fft.fft2(self._input_images[0]), dtype=np.complex128)
        self._mixed_image = None

    @property 
    def input_images(self):
        return self._input_images
    @input_images.setter
    def input_images(self,value):
        self._input_images = value
        
    @property 
    def mixed_fft(self):
        return self._mixed_fft
    @mixed_fft.setter
    def mixed_fft(self,value):
        self._mixed_fft = value
    @property 
    def mixed_image(self):
        return self._mixed_image
    @mixed_image.setter
    def mixed_image(self,value):
        self._mixed_image = value


    # def customize_images_size(self):
    #     small_size_image = (100000, 100000)
    #     for i in range(4):
    #         if self._input_images[i].image_size < small_size_image
    #             small_size_image = self._input_images[i].image_size
    #     print(small_size_image)
    def mix_images(self):
        for i in range(4):
            weighted_fft = (
                    self.input_images[i].weight_of_magnitude * np.abs(self.input_images[i].image_fft)
                    * np.exp(1j * self.input_images[i].weight_of_phase)
            )
            self._mixed_fft += weighted_fft
        self.mixed_image = np.abs(np.fft.ifft2(self._mixed_fft)).astype(np.uint8)
        return self.mixed_image


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
