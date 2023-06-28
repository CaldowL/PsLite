import math
import os

import cv2
import numpy as np


def image_zip_write(width: int, height: int, kernel: int, file_name: str, arr_img: list, arr_gray: list):
    """
    对图像压缩，将比特面，灰度值等图像信息全部打包为单个压缩文件
    :param width: 图片宽度
    :param height: 图片高度
    :param kernel: 方块大小
    :param file_name: 需要保存的文件名
    :param arr_img: 图片比特面数据
    :param arr_gray: 图像灰度值数据
    :return: None
    """
    if os.path.exists(file_name):
        os.remove(file_name)

    arr_img = arr_img[:len(arr_img) - len(arr_img) % 8]
    s_img = ["".join(["1" if j == 1 else "0" for j in arr_img[i:i + 8]]) for i in range(0, len(arr_img), 8)]

    with open(file_name, "wb+") as f:
        w = bin(width).replace("0b", "").zfill(16)
        h = bin(height).replace("0b", "").zfill(16)
        k = bin(kernel).replace("0b", "").zfill(8)

        f.write(bytearray([int(w[:8], 2)]))
        f.write(bytearray([int(w[8:], 2)]))
        f.write(bytearray([int(h[:8], 2)]))
        f.write(bytearray([int(h[8:], 2)]))
        f.write(bytearray([int(k, 2)]))

        for s in s_img:
            f.write(bytearray([int(s, 2)]))

        for s in arr_gray:
            f.write(bytearray([s]))


def image_encode_square(image: np.ndarray, kernel: int, file_name: str = "output.dat"):
    """
    图像压缩，获取图像数据
    :param image: cv读入的图片
    :param kernel: 方块核大小
    :param file_name: 保存的图片路径
    :return:
    """
    image = image[:, :, 0]
    rows, cols = image.shape[:2]
    rows -= rows % kernel
    cols -= cols % kernel

    filtered_image = np.zeros((rows, cols))
    gray_rebuild = []
    for i in range(0, rows, kernel):
        for j in range(0, cols, kernel):
            window = image[i:i + kernel, j: j + kernel]

            ave = np.average(window)
            st = math.sqrt(np.mean(np.square(window - np.mean(window))))

            window = window >= ave
            filtered_image[i:i + kernel, j: j + kernel] = window

            count_1 = np.count_nonzero(window)
            count_0 = kernel ** 2 - count_1
            if count_0 == 0 or count_1 == 0:
                a0 = ave
                a1 = ave
            else:
                a0 = int(ave - st * math.sqrt(count_1 / count_0))
                a1 = int(ave + st * math.sqrt(count_0 / count_1))

            gray_rebuild.append(a0)
            gray_rebuild.append(a1)

    filtered_image = filtered_image.flatten().tolist()
    filtered_image = list(map(int, filtered_image))

    image_zip_write(rows, cols, kernel, file_name, filtered_image, gray_rebuild)


def image_unzip_read(file_name):
    """
    按字节读入文件
    :param file_name: 读入的文件名
    :return: 宽度，高度，方块大小，比特面，重建灰度值
    """
    with open(file_name, "rb") as file:
        byte_data = file.read(2)
        s = "".join([bin(i)[2:].zfill(8) for i in byte_data])
        rows = int(s, 2)

        byte_data = file.read(2)
        s = "".join([bin(i)[2:].zfill(8) for i in byte_data])
        cols = int(s, 2)

        byte_data = file.read(1)
        s = "".join([bin(i)[2:].zfill(8) for i in byte_data])
        kernel = int(s, 2)

        image = ""
        byte_data = file.read(int(rows * cols / 8))
        for byte in byte_data:
            byte_str = bin(byte)[2:].zfill(8)
            image += byte_str

        gray = []
        byte_data = file.read()
        for byte in byte_data:
            byte_str = bin(byte)[2:].zfill(8)
            gray.append(int(byte_str, 2))

    return rows, cols, kernel, [int(i) for i in image], gray


def image_decode_square(file_name, out_file_name):
    """
    图片解码
    :param file_name:需要解码的文件名
    :param out_file_name:输出的文件名
    :return:None
    """
    assert file_name.split(".")[-1] == "dat", "文件类型选择错误"
    r, c, k, img, g = image_unzip_read(file_name)
    index_gray = 0
    img = np.reshape(img, (r, c))
    for i in range(0, r, k):
        for j in range(0, c, k):
            window = img[i:i + k, j:j + k]
            a0 = g[index_gray]
            a1 = g[index_gray + 1]
            index_gray += 2
            window = np.where(window == 1, a1, a0)
            img[i:i + k, j:j + k] = window
    cv2.imwrite(out_file_name, img)


if __name__ == '__main__':
    img = cv2.imread("imgs/HOUSE1.BMP")
    image_encode_square(img, 4, "output.dat")
    print("压缩完成")
    image_decode_square("output.dat", "out.bmp")
    print("解压完成")
