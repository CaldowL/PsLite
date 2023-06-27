import math
from copy import copy, deepcopy

import PyCameraList.camera_device
import cv2
import json

import numpy as np
import requests
import numpy
import time
from PyQt5.QtGui import QImage, QPixmap

import model


def get_camera_list():
    cameras = dict(PyCameraList.camera_device.list_video_devices())
    res = []
    for i in cameras.keys():
        res.append(f"{cameras[i]}:{i}")
    return res


def img_to_pixmap(img: np.ndarray, filp=False):
    """
    将cv读取的图片转为PixMap
    :param img:
    :return:
    """
    shrink = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if filp:
        shrink = cv2.flip(shrink, 1)
    # cv 图片转换成 qt图片
    qt_img = QImage(shrink.data,  # 数据源
                    shrink.shape[1],  # 宽度
                    shrink.shape[0],  # 高度
                    shrink.shape[1] * 3,  # 行字节数
                    QImage.Format_RGB888)

    return QPixmap().fromImage(qt_img)


def zfill(_list: list, lenght):
    """
    在list前面补0
    :param _list:
    :param lenght:
    :return:
    """
    if len(_list) < lenght:
        _list = _list[::-1]
        for i in range(lenght - len(_list)):
            _list.append(0)
        _list = _list[::-1]
    return _list


def crop_image(image, x1, y1, x2, y2):
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    cropped_image = image[y1:y2, x1:x2]
    return cropped_image


def rgbtohsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    B, G, R = cv2.split(rgb_lwpImg)
    # 归一化到[0,1]
    B = B / 255.0
    G = G / 255.0
    R = R / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv2.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((R[i, j] - G[i, j]) + (R[i, j] - B[i, j]))
            den = np.sqrt((R[i, j] - G[i, j]) ** 2 + (R[i, j] - B[i, j]) * (G[i, j] - B[i, j]))
            theta = float(np.arccos(num / den))

            if den == 0:
                H = 0
            elif B[i, j] <= G[i, j]:
                H = theta
            else:
                H = 2 * 3.14169265 - theta

            min_RGB = min(min(B[i, j], G[i, j]), R[i, j])
            sum = B[i, j] + G[i, j] + R[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / sum

            H = H / (2 * 3.14159265)
            I = sum / 3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H * 255
            hsi_lwpImg[i, j, 1] = S * 255
            hsi_lwpImg[i, j, 2] = I * 255
    return hsi_lwpImg


def hsitorgb(hsi_img):
    h = int(hsi_img.shape[0])
    w = int(hsi_img.shape[1])
    H, S, I = cv2.split(hsi_img)
    H = H / 255.0
    S = S / 255.0
    I = I / 255.0
    bgr_img = hsi_img.copy()
    B, G, R = cv2.split(bgr_img)
    for i in range(h):
        for j in range(w):
            if S[i, j] < 1e-6:
                R = I[i, j]
                G = I[i, j]
                B = I[i, j]
            else:
                H[i, j] *= 360
                if H[i, j] > 0 and H[i, j] <= 120:
                    B = I[i, j] * (1 - S[i, j])
                    R = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j] * math.pi / 180)) / math.cos(
                        (60 - H[i, j]) * math.pi / 180))
                    G = 3 * I[i, j] - (R + B)
                elif H[i, j] > 120 and H[i, j] <= 240:
                    H[i, j] = H[i, j] - 120
                    R = I[i, j] * (1 - S[i, j])
                    G = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j] * math.pi / 180)) / math.cos(
                        (60 - H[i, j]) * math.pi / 180))
                    B = 3 * I[i, j] - (R + G)
                elif H[i, j] > 240 and H[i, j] <= 360:
                    H[i, j] = H[i, j] - 240
                    G = I[i, j] * (1 - S[i, j])
                    B = I[i, j] * (1 + (S[i, j] * math.cos(H[i, j] * math.pi / 180)) / math.cos(
                        (60 - H[i, j]) * math.pi / 180))
                    R = 3 * I[i, j] - (G + B)
            bgr_img[i, j, 0] = int(B * 255)
            bgr_img[i, j, 1] = int(G * 255)
            bgr_img[i, j, 2] = int(R * 255)
    return bgr_img
