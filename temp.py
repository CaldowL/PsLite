import math
import os

import cv2
import numpy as np


def image_encode_square(image: np.ndarray, kernel: int):
    image = image[:, :, 0]
    rows, cols = image.shape[:2]
    rows -= rows % kernel
    cols -= cols % kernel

    filtered_image = np.zeros((rows, cols))
    for i in range(0, rows, kernel):
        for j in range(0, cols, kernel):
            window = image[i:i + kernel, j: j + kernel]

            ave = np.average(window)
            st = math.sqrt(np.mean(np.square(window - np.mean(window))))

            # print(window, ave)
            window = window >= ave
            filtered_image[i:i + kernel, j: j + kernel] = window

            count_1 = np.count_nonzero(window)
            count_0 = kernel ** 2 - count_1

            a0 = int(ave - st * math.sqrt(count_1 / count_0))
            a1 = int(ave + st * math.sqrt(count_0 / count_1))

    filtered_image = filtered_image.flatten()
    filtered_image = filtered_image.tolist()
    filtered_image = list(map(int, filtered_image))

    image_zip(rows, cols, kernel, "output.bin", filtered_image)


def image_zip(width: int, height: int, kernel: int, file_name: str, arr: list):
    if os.path.exists(file_name):
        os.remove(file_name)

    arr = arr[:len(arr) - len(arr) % 8]
    ss = ["".join(["1" if j == 1 else "0" for j in arr[i:i + 8]]) for i in range(0, len(arr), 8)]

    with open(file_name, "wb+") as f:
        w = bin(width).replace("0b", "").zfill(16)
        h = bin(height).replace("0b", "").zfill(16)
        k = bin(kernel).replace("0b", "").zfill(8)
        print(width, w)
        print(height, h)
        print(kernel, k)

        f.write(bytearray([int(w[:8], 2)]))
        f.write(bytearray([int(w[8:], 2)]))
        f.write(bytearray([int(h[:8], 2)]))
        f.write(bytearray([int(h[8:], 2)]))
        f.write(bytearray([int(k, 2)]))

        for s in ss:
            byte_val = int(s, 2)
            f.write(bytearray([byte_val]))


def image_read(file_name):
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

        byte_data = file.read()  # 读取整个文件为字节数据
        image = ""
        bit_sequence = ""
        for byte in byte_data:
            byte_str = bin(byte)[2:].zfill(8)  # 将每个字节转换为二进制字符串

            for bit in byte_str:
                bit_sequence += bit  # 将每个位添加到序列中

                if len(bit_sequence) == 8:
                    print(bit_sequence)
                    image += bit_sequence
                    bit_sequence = ""  # 重新开始一个新的序列
    return rows, cols, kernel, [int(i) for i in image]


img = cv2.imread("imgs/HOUSE1.BMP")
image_encode_square(img, 4)
