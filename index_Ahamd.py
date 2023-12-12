from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
from PyQt5.uic import loadUiType

ui, _ = loadUiType('main.ui')

class BrightnessContrastGraphicsView(QGraphicsView):
    def __init__(self, image_path):
        super().__init__()

        # Load an image and create a QGraphicsPixmapItem to display it
        self.scene = QGraphicsScene(self)
        # self.pixmap_item = QGraphicsPixmapItem(QPixmap(image_path))
        # self.scene.addItem(self.pixmap_item)
        self.setScene(self.scene)
        self.original_pixmap = QPixmap(image_path)
        self.grayscale_pixmap = self.convert_to_grayscale(self.original_pixmap)
        self.pixmap_item = QGraphicsPixmapItem(self.grayscale_pixmap)
        self.scene.addItem(self.pixmap_item)

        # Enable the ability to move the image with the mouse
        # self.setDragMode(QGraphicsView.ScrollHandDrag)

        # Variables to track the rectangle drawing
        self.drawing_rectangle = False
        self.rectangle_item = None
        self.start_point = None
    def convert_to_grayscale(self, pixmap):
        image = pixmap.toImage().convertToFormat(QImage.Format_Grayscale8)
        grayscale_pixmap = QPixmap.fromImage(image)
        return grayscale_pixmap

    # def mouseMoveEvent(self, event: QMouseEvent):
    #     # You can capture the mouse position during the drag here
    #     print("Mouse position during drag:", event.pos())
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.start_point = event.pos()
            self.drawing_rectangle = True

    def mouseMoveEvent(self, event):
        if event.button() != Qt.RightButton:
            if self.drawing_rectangle:
                current_point = event.pos()
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
            print("Mouse position during drag:", event.pos())

    def mouseReleaseEvent(self, event):
        if self.drawing_rectangle:
            self.drawing_rectangle = False
            self.start_point = None

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


class MainApp(QWidget, ui):
    _show_hide_flag = True

    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.resize(1450, 900)

        # Load the image and display it in the QGraphicsView
        image_path = 'download.jpeg'
        self.graphics_view = BrightnessContrastGraphicsView(image_path)
        # self.original_gray_image_1.setDragMode(QGraphicsView.ScrollHandDrag)
        # self.scene = QGraphicsScene(self)
        # self.pixmap_item = QGraphicsPixmapItem(QPixmap(image_path))
        # self.scene.addItem(self.pixmap_item)
        # self.original_gray_image_1.setScene(self.scene)


        self.graphics_view_layout1 = QHBoxLayout(self.original_gray_image_1)
        self.graphics_view_layout1.addWidget(self.graphics_view)
        self.original_gray_image_1.setLayout(self.graphics_view_layout1)

def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
