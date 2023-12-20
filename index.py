from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
from PyQt5.uic import loadUiType
import cv2
from image_mixer import *

ui, _ = loadUiType('main.ui')


class MainApp(QWidget, ui):
    _show_hide_flag = True

    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.resize(1450, 900)

        self.image_1 = ViewOriginal()
        # self.image_1.mouseMoveEvent(QMouseEvent)

        self.graphics_view_layout1 = QHBoxLayout(self.graphicsView_original_1)
        self.graphics_view_layout1.addWidget(self.image_1)
        self.graphicsView_original_1.setLayout(self.graphics_view_layout1)


        # self.image_2 = ImageViewer('images (1).jpeg')
        # self.image_2.image_size = (250, 210)
        # qt_image = self.convert_cv_to_qt(self.image_2.gray_scale_image)
        # scene = QGraphicsScene(self)
        # pixmap_item = QGraphicsPixmapItem(qt_image)
        # scene.addItem(pixmap_item)
        # self.graphicsView_original_2.setScene(scene)
        #
        # self.image_3 = ImageViewer('download.jpeg')
        # self.image_3.image_size = (250, 210)
        # qt_image = self.convert_cv_to_qt(self.image_3.gray_scale_image)
        # scene = QGraphicsScene(self)
        # pixmap_item = QGraphicsPixmapItem(qt_image)
        # scene.addItem(pixmap_item)
        # self.graphicsView_original_3.setScene(scene)
        #
        # self.image_4 = ImageViewer('download (1).jpeg')
        # self.image_4.image_size = (250, 210)
        # qt_image = self.convert_cv_to_qt(self.image_4.gray_scale_image.astype(np.uint8))
        # scene = QGraphicsScene(self)
        # pixmap_item = QGraphicsPixmapItem(qt_image)
        # scene.addItem(pixmap_item)
        # self.graphicsView_original_4.setScene(scene)

        # self.image_9 = ImageViewer('images (1).jpeg')
        # self.image_9.original_image_magnitude
        # qt_image = self.convert_cv_to_qt(self.image_9.original_image_magnitude.astype(np.uint8))
        # scene = QGraphicsScene(self)
        # pixmap_item = QGraphicsPixmapItem(qt_image)
        # scene.addItem(pixmap_item)
        # self.graphicsView_weight_1.setScene(scene)



        self.image_w = ViewWeight("images.jpeg")
        self.graphics_view_layout1 = QHBoxLayout(self.graphicsView_weight_1)
        self.graphics_view_layout1.addWidget(self.image_w)
        self.graphicsView_weight_1.setLayout(self.graphics_view_layout1)

    def convert_cv_to_qt(self, cv_image):
        height, width = cv_image.shape
        bytes_per_line = width
        qt_image = QImage(cv_image.data.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qt_image)
    # def weights(self):
    #     path1=self.image_1.path()
    #     # print(path1)
    #     self.image_w = ViewWeight(path1)
    #     self.graphics_view_layout1 = QHBoxLayout(self.graphicsView_weight_1)
    #     self.graphics_view_layout1.addWidget(self.image_w)
    #     self.graphicsView_weight_1.setLayout(self.graphics_view_layout1)

        # self.graphicsView_7 = CustomGraphicsView(self.graphicsView_7, self)

        # image_path = 'download.jpeg'
        # scene = QGraphicsScene(self)
        # pixmap = QPixmap(image_path)
        # item = QGraphicsPixmapItem(pixmap)
        # scene.addItem(item)
        # self.graphicsView_7.setScene(scene)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()








class BrightnessContrastGraphicsView(QGraphicsView):
    def __init__(self, image_path):
        super().__init__()



        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.original_pixmap = QPixmap(image_path)
        self.grayscale_pixmap = self.convert_to_grayscale(self.original_pixmap)
        self.pixmap_item = QGraphicsPixmapItem(self.grayscale_pixmap)
        self.scene.addItem(self.pixmap_item)

        self.drawing_rectangle = False
        self.rectangle_item = None
        self.start_point = None
    def convert_to_grayscale(self, pixmap):
        image = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        grayscale_pixmap = QPixmap.fromImage(image)
        return grayscale_pixmap

    def mouseMoveEvent(self, event: QMouseEvent):
        # You can capture the mouse position during the drag here
        print("Mouse position during drag:", self.mapToScene(event.pos()))
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.start_point = self.mapToScene(event.pos())
            self.drawing_rectangle = True

    def mouseMoveEvent(self, event):
        if event.button() != Qt.RightButton:
            if self.drawing_rectangle:
                current_point =self.mapToScene(event.pos())
                rect = QRectF(self.start_point, current_point).normalized()
                
                if not self.rectangle_item:
                    self.rectangle_item = QGraphicsRectItem(rect)
                    pen = QPen(QColor(Qt.white), 2,
                               Qt.DashLine)  # You can use Qt.DashLine or Qt.DotLine for a dashed or dotted line
                    self.rectangle_item.setPen(pen)
                    self.scene.addItem(self.rectangle_item)
                else:
                    self.rectangle_item.setRect(rect)
        else:
            print("Mouse position during drag:", self.mapToScene(event.pos()))

    def mouseReleaseEvent(self, event):
        if self.drawing_rectangle:
            self.drawing_rectangle = False
            self.start_point = None


class CustomGraphicsView(QGraphicsView):
    def __init__(self, graphics_view, parent=None):
        super(CustomGraphicsView, self).__init__(parent)
        self.graphics_view = graphics_view
        self.setup_graphics_view()

        self.rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()

        # Additional attributes for drawing rectangles
        self.drawing_rectangle = False
        self.start_point = None
        self.rectangle_item = None

    def setup_graphics_view(self):
        image_path = 'download.jpeg'
        scene = QGraphicsScene(self)
        pixmap = QPixmap(image_path)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphics_view.setScene(scene)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
        
            self.origin =self.mapToScene(event.pos())
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

            # Start drawing a rectangle
            self.drawing_rectangle = True
            self.start_point = self.mapToScene(event.pos())

    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, self.mapToScene(event.pos())).normalized())

        if self.drawing_rectangle:
            current_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, current_point).normalized()
            
            if not self.rectangle_item:
                self.rectangle_item = QGraphicsRectItem(rect)
                self.rectangle_item.setPen(QColor(Qt.blue))
                self.scene().addItem(self.rectangle_item)
            else:
                self.rectangle_item.setRect(rect)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            rect = self.rubber_band.geometry()
            selected_region = self.mapToScene(rect).boundingRect()
            print("Selected Region:", selected_region)


            # Get information about the points inside the rectangle
            points_in_rectangle = self.get_points_in_rectangle(self.rectangle_item.rect())
            print("Points inside the rectangle:", points_in_rectangle)

    def get_points_in_rectangle(self, rectangle):
        # Get the region of interest from the image
        region_of_interest = self.grayscale_pixmap.toImage().copy(rectangle.toRect()).convertToFormat(QImage.Format_Grayscale8)
        # Get the pixel values within the rectangle
        points = []
        for y in range(region_of_interest.height()):
            for x in range(region_of_interest.width()):
                pixel_value = region_of_interest.pixel(x, y)
                points.append((x + int(rectangle.x()), y + int(rectangle.y()), QColor(pixel_value).getRgb()))

        return points


        # Reset drawing variables
        self.drawing_rectangle = False
        self.start_point = None
        self.rectangle_item = None

    def get_points_in_rectangle(self, rect):
        # Implement your logic to obtain indices or points inside the rectangle here
        # For now, returning a dummy result
        return "Not implemented yet"
