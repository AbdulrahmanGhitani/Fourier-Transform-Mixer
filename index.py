from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
from PyQt5.uic import loadUiType

ui, _ = loadUiType('main.ui')


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
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()

            # Start drawing a rectangle
            self.drawing_rectangle = True
            self.start_point = event.pos()

    def mouseMoveEvent(self, event):
        if not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

        if self.drawing_rectangle:
            current_point = event.pos()
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

            # Reset drawing variables
            self.drawing_rectangle = False
            self.start_point = None
            self.rectangle_item = None

    def get_points_in_rectangle(self, rect):
        # Implement your logic to obtain indices or points inside the rectangle here
        # For now, returning a dummy result
        return "Not implemented yet"


class MainApp(QWidget, ui):
    _show_hide_flag = True

    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.resize(1450, 900)


        # self.graphicsView_7 = CustomGraphicsView(self.graphicsView_7, self)

        image_path = 'download.jpeg'
        scene = QGraphicsScene(self)
        pixmap = QPixmap(image_path)
        item = QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        self.graphicsView_7.setScene(scene)


def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
