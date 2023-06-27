# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1465, 971)
        self.groupBox = QtWidgets.QGroupBox(Form)
        self.groupBox.setGeometry(QtCore.QRect(1120, 10, 341, 961))
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 40, 121, 61))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 210, 161, 61))
        self.pushButton_2.setObjectName("pushButton_2")
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setGeometry(QtCore.QRect(180, 230, 111, 22))
        self.checkBox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBox.setObjectName("checkBox")
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 130, 161, 61))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_4.setGeometry(QtCore.QRect(10, 290, 161, 61))
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalSlider = QtWidgets.QSlider(self.groupBox)
        self.horizontalSlider.setGeometry(QtCore.QRect(10, 400, 311, 22))
        self.horizontalSlider.setMinimum(3)
        self.horizontalSlider.setMaximum(15)
        self.horizontalSlider.setSingleStep(2)
        self.horizontalSlider.setPageStep(2)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setTickInterval(1)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(90, 440, 171, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 440, 51, 61))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalSlider_2 = QtWidgets.QSlider(self.groupBox)
        self.horizontalSlider_2.setGeometry(QtCore.QRect(10, 360, 311, 22))
        self.horizontalSlider_2.setMinimum(3)
        self.horizontalSlider_2.setMaximum(15)
        self.horizontalSlider_2.setSingleStep(2)
        self.horizontalSlider_2.setPageStep(2)
        self.horizontalSlider_2.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_2.setTickInterval(1)
        self.horizontalSlider_2.setObjectName("horizontalSlider_2")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(180, 300, 151, 41))
        self.comboBox.setObjectName("comboBox")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(10, 130, 531, 541))
        self.label.setStyleSheet("background-color:#dadada;")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(580, 130, 531, 541))
        self.label_4.setStyleSheet("background-color:#dadada;")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(0, 930, 1121, 41))
        self.label_5.setStyleSheet("background-color:#dadada")
        self.label_5.setObjectName("label_5")

        self.retranslateUi(Form)
        self.pushButton.clicked.connect(Form.click_select_image) # type: ignore
        self.pushButton_2.clicked.connect(Form.img_equalize) # type: ignore
        self.pushButton_3.clicked.connect(Form.gray_reversal) # type: ignore
        self.pushButton_4.clicked.connect(Form.median_filter) # type: ignore
        self.horizontalSlider.sliderMoved['int'].connect(Form.setValue2) # type: ignore
        self.horizontalSlider_2.sliderMoved['int'].connect(Form.setValue) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox.setTitle(_translate("Form", "工作区"))
        self.pushButton.setText(_translate("Form", "选择图片"))
        self.pushButton_2.setText(_translate("Form", "直方图均衡"))
        self.checkBox.setText(_translate("Form", "显示过程"))
        self.pushButton_3.setText(_translate("Form", "灰度反转"))
        self.pushButton_4.setText(_translate("Form", "中值滤波"))
        self.label_2.setText(_translate("Form", "(3,3)"))
        self.label_3.setText(_translate("Form", "核:"))
        self.label.setText(_translate("Form", "原图预览"))
        self.label_4.setText(_translate("Form", "效果图预览"))
        self.label_5.setText(_translate("Form", "状态栏"))
