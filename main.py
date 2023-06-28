import logging
import sys
import threading

import traceback

import win32api
import win32con
import win32gui
from PyQt5 import QtCore

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QFileDialog

import FuncsAPI
import Funcs
import Utils
import untitled
from Utils import *


def show_exception(e):
    hwnd = win32gui.GetDesktopWindow()
    win32api.MessageBox(hwnd, str(e), "警告", win32con.MB_OK | win32con.MB_ICONERROR)


def handle_exception(exc_type, exc_value, exc_traceback):
    print(f"Exception Type: {exc_type}")
    print(f"Exception Value: {exc_value}")
    print(f"Traceback: {exc_traceback}")
    show_exception(exc_value)


class ThreadWithException(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, *, daemon=None):
        super().__init__()
        self.target = target
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.exception = None

    def run(self):
        try:
            if self.target:
                self.target(*self.args, **self.kwargs)
        except Exception as e:
            msg = traceback.format_exc()
            print(msg)
            show_exception(str(e))
        finally:
            del self.target, self.args, self.kwargs


class Form(QMainWindow, untitled.Ui_Form):
    def __init__(self):
        super(Form, self).__init__()
        self.setupUi(self)

        self.pushButton_3.setVisible(False)

        self.comboBox.addItems(["矩形框", "菱形框", "圆形框", "自适应滤波"])
        self.set_status("初始化完成")

    def set_status(self, status):
        self.label_5.setText(f"  {status}")

    def click_select_image(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图片(*.png *.jpg *.bmp)")
            assert file_path != ""
            print(file_path)
            model.file_select_path = file_path
        except Exception as e:
            print(e)
            self.set_status("图片选择失败")
            return

        img = cv2.imread(file_path)
        model.file_select_ndarray = img
        self.label.setScaledContents(True)
        self.label_4.setScaledContents(True)

        self.label.setPixmap(Utils.img_to_pixmap(img))

        self.set_status("图片选择完成")

    def img_equalize(self):
        assert model.file_select_ndarray is not None, "请先选择图片"
        color = False
        print(model.file_select_ndarray.shape)

        for _ in range(min(150, min(model.file_select_ndarray.shape[0], model.file_select_ndarray.shape[1]))):
            if len(set(model.file_select_ndarray[_][_])) == 1:
                color = False
                self.set_status("开始灰度图像直方图均衡")
            else:
                color = True
                self.set_status("开始彩色图像直方图均衡")
                break

        print("灰色图像") if not color else print("彩色图像")

        ThreadWithException(target=FuncsAPI.api_histogram_equalization,
                            args=(self, self.checkBox.checkState(), color,)).start()

    def median_filter(self):
        ThreadWithException(target=FuncsAPI.api_median_filter,
                            args=(
                                self, (model.slider_value1, model.slider_value2,),
                                self.comboBox.currentText(),)).start()

    def gray_reversal(self):
        if model.file_select_path == "":
            return

    def square_zip(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "图片(*.png *.jpg *.bmp)")
        assert file_path != "", "请选择文件"
        ThreadWithException()

    def square_unzip(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择压缩文件", "", "压缩文件(*.dat)")
        assert file_path != "", "请选择文件"

    def setValue(self, value):
        model.slider_value1 = value if value % 2 != 0 else value - 1

        self.label_2.setText(f"({model.slider_value1},{model.slider_value2})")
        self.set_status("选择中值滤波核大小:" + f"({model.slider_value1},{model.slider_value2})")

    def setValue2(self, value):
        model.slider_value2 = value if value % 2 != 0 else value - 1
        self.label_2.setText(f"({model.slider_value1},{model.slider_value2})")
        self.set_status("选择中值滤波核大小:" + f"({model.slider_value1},{model.slider_value2})")


if __name__ == '__main__':
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    sys.excepthook = handle_exception
    loger = logging.Logger("log")
    loger.setLevel(logging.INFO)

    app = QApplication(sys.argv)
    MainWindow = Form()
    MainWindow.show()
    sys.exit(app.exec_())
