from PyQt5.QtQuick import QQuickView
from PyQt5.QtGui import QGuiApplication, QWindow
from PyQt5.QtCore import QObject, QUrl, pyqtProperty, pyqtSignal
import os

class DenoiserResults(QObject):
    metricsChanged = pyqtSignal()
    imageChanged = pyqtSignal()

    def __init__(self, parent, name, data):
        super().__init__(parent)
        self._name = name
        self._data = data
        self._image = None

    @pyqtProperty(str, constant=True)
    def name(self):
        return self._name if self._name != "none" else "noisy"

    @pyqtProperty('QVariantMap', notify=metricsChanged)
    def metrics(self):
        print(self._data[self._data.image == self._image])
        return {"foo": 1}

    @pyqtProperty(str, notify=imageChanged)
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value
        self.imageChanged.emit()
        self.metricsChanged.emit()


class Viewer(QObject):
    def __init__(self, args, image_path, data):
        super().__init__()
        self._data = data
        self._image_path = image_path


        current_dir = os.path.abspath(os.path.dirname(__file__))

        # configure the view and load the QML file
        self.app = QGuiApplication(args)
        self.view = QQuickView()

        # create the denoiser list before setting it as a context property
        self.createDenoiserList()

        self.view.rootContext().setContextProperty("result_data", self)
        self.view.setResizeMode(QQuickView.SizeRootObjectToView)
        self.view.setSource(QUrl.fromLocalFile(os.path.join(current_dir, "viewer.qml")))

    @pyqtProperty(list, constant=True)
    def image_names(self):
        return self._data.image.unique().tolist()

    @pyqtProperty(str, constant=True)
    def image_path(self):
        return self._image_path

    @pyqtProperty(list, constant=True)
    def denoisers(self):
        print(len(self._denoisers))
        return self._denoisers

    def createDenoiserList(self):
        print(self._data.denoiser.unique())
        print(self._data[self._data.denoiser == "none"])
        self._denoisers = [DenoiserResults(self, name, self._data[self._data.denoiser == name]) for name in self._data.denoiser.unique()]
        print([self._data[self._data.denoiser == name] for name in self._data.denoiser.unique()])

    def run(self):
        self.view.showMaximized()
        self.app.exec_()

