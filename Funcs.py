import math
import os

import cv2
import numpy as np


def origin_histogram(img):
    histogram = {}
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            k = img[i][j]
            if k in histogram.keys():
                histogram[k] += 1
            else:
                histogram[k] = 1
    sorted_list = sorted(histogram.items(), key=lambda x: x[0])  # 根据灰度值进行从低至高的排序
    return dict(sorted_list)


def equalization_histogram(histogram, img):
    pr = {}  # 建立概率分布映射表
    for i in histogram.keys():
        pr[i] = histogram[i] / (img.shape[0] * img.shape[1])

    tmp = 0
    for m in pr.keys():
        tmp += pr[m]
        pr[m] = max(histogram) * tmp
    new_img = np.zeros(shape=(img.shape[0], img.shape[1]), dtype=np.uint8)
    for k in range(img.shape[0]):
        for l in range(img.shape[1]):
            new_img[k][l] = pr[img[k][l]]
    return new_img


def GrayHist(img):
    height, width = img.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(height):
        for j in range(width):
            grayHist[img[i][j]] += 1

    return [i / (height * width) for i in grayHist]


def median_filter(image, kernel_size=3):
    rows, cols = image.shape[:2]
    filtered_image = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            window = image[max(0, i - kernel_size):min(rows, i + kernel_size + 1),
                     max(0, j - kernel_size):min(cols, j + kernel_size + 1)]
            filtered_image[i, j] = np.median(window, axis=(0, 1))

    return filtered_image


def median_filter_square(image, kernel_size=(3, 3)):
    """
    中值滤波，矩形框
    :param image:
    :param kernel_size:
    :return:
    """
    rows, cols = image.shape[:2]
    filtered_image = np.zeros_like(image)
    kernel_rows, kernel_cols = kernel_size

    for i in range(rows):
        for j in range(cols):
            start_row = max(0, i - kernel_rows // 2)
            end_row = min(rows, i + kernel_rows // 2 + 1)
            start_col = max(0, j - kernel_cols // 2)
            end_col = min(cols, j + kernel_cols // 2 + 1)

            window = image[start_row:end_row, start_col:end_col]
            filtered_image[i, j] = np.median(window, axis=(0, 1))

    return filtered_image


def median_filter_circle2(image, kernel_r):
    """

    :param image:
    :param kernel_r:
    :return:
    """
    rows, cols = image.shape[:2]
    filtered_image = np.zeros_like(image)

    for i in range(rows):
        for j in range(cols):
            start_row = max(0, i - kernel_r)
            end_row = min(rows, i + kernel_r + 1)
            start_col = max(0, j - kernel_r)
            end_col = min(cols, j + kernel_r + 1)

            window = image[start_row:end_row, start_col:end_col]

            pad_xf = 0 if start_row > 0 else kernel_r - start_row
            pad_xb = 0 if end_row < rows else rows - end_row
            pad_yf = 0 if start_col > 0 else kernel_r - start_col
            pad_yb = 0 if end_col < cols else cols - end_col

            top, bottom, left, right = pad_yf, pad_yb, pad_xf, pad_xb
            h, w, _ = window.shape
            new_img = np.ones((h + top + bottom, w + left + right, 3), dtype=np.uint8) * 255
            new_img[top:h + top, left:w + left] = window
            window = new_img

            circle_center = [int(window.shape[0] // 2), int(window.shape[1] // 2)]

            # 原始循环操作
            # window_circle = []
            # for m in range(window.shape[0]):
            #     for n in range(window.shape[1]):
            #         if abs(circle_center[0] - m) ** 2 + abs(circle_center[1] - n) ** 2 <= kernel_r ** 2:
            #             window_circle.append(window[m][n])

            # 列表生成式
            # window_circle = [window[m][n] for m in range(window.shape[0]) for n in range(window.shape[1]) if
            #                  abs(circle_center[0] - m) ** 2 + abs(circle_center[1] - n) ** 2 <= kernel_r ** 2]

            # 矩阵运算操作
            x, y = np.indices(window.shape[:2])
            dist = np.sqrt((x - circle_center[0]) ** 2 + (y - circle_center[1]) ** 2)
            window_circle = window[dist <= kernel_r]
            filtered_image[i, j] = np.median(window_circle, axis=0)

    return filtered_image


def median_filter_circle(image, kernel_r):
    """
    中值滤波，圆形掩膜
    :param image: np.ndarray
    :param kernel_r: int
    :return: np.ndarray
    """
    rows, cols = image.shape[:2]
    filtered_image = np.zeros_like(image)

    pad_size = kernel_r
    padded_img = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

    # 创建圆形掩膜矩阵
    i, j = np.indices((kernel_r * 2 + 1, kernel_r * 2 + 1))
    mask = (i - kernel_r) ** 2 + (j - kernel_r) ** 2 <= kernel_r ** 2

    for i in range(rows):
        for j in range(cols):
            window = padded_img[i:i + (kernel_r * 2 + 1), j:j + (kernel_r * 2 + 1)]
            window_circle = window[mask]
            filtered_image[i, j] = np.median(window_circle, axis=0)

    return filtered_image


def median_filter_diamond(image, kernel_r):
    """
    中值滤波，菱形框
    :param image: np.ndarray
    :param kernel_r: int
    :return: np.ndarray
    """
    rows, cols = image.shape[:2]
    filtered = np.zeros_like(image)

    # 扩展图像边缘
    pad_size = kernel_r
    padded_img = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

    i, j = np.indices((kernel_r * 2 + 1, kernel_r * 2 + 1), dtype=np.uint8)
    mask = i + j >= kernel_r  # 创建 右上角那部分的掩膜
    mask = np.logical_not(
        np.logical_xor(np.flip(np.logical_xor(np.flip(mask), mask), axis=1), np.logical_xor(np.flip(mask), mask)))

    for i in range(rows):
        for j in range(cols):
            window = padded_img[i:i + (kernel_r * 2 + 1), j:j + (kernel_r * 2 + 1)]
            window = window[mask]
            filtered[i, j] = np.median(window, axis=0)

    return filtered


def adaptive_median_filter(image, window_size, max_window_size):
    height, width, _ = image.shape
    filtered_image = np.copy(image)

    pad_size = window_size // 2

    for i in range(pad_size, height - pad_size):
        for j in range(pad_size, width - pad_size):
            window = image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            window_flat = window.flatten()

            while True:
                median = np.median(window_flat)
                max_value = np.max(window_flat)
                min_value = np.min(window_flat)

                if min_value < median < max_value:
                    if image[i, j] < min_value or image[i, j] > max_value:
                        filtered_image[i, j] = median
                    break

                window_size += 2
                if window_size <= max_window_size:
                    pad_size = window_size // 2
                    window = image[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
                    window_flat = window.flatten()
                else:
                    filtered_image[i, j] = median
                    break

    return filtered_image


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
    img = cv2.imread("imgs/a.png")
    filtered_image = adaptive_median_filter(img, 3, 7)

    cv2.imshow("Original Image", img)
    cv2.imshow("Filtered Image", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # img = cv2.imread("imgs/a.png")
    # # 读取原始图像
    # img = cv2.imread('E:/Projects/Python/PsLite/imgs/a.png', cv2.COLOR_BGR2HSV)
    # # 计算原图灰度直方图
    # _origin_histogram = origin_histogram(img)
    # print(_origin_histogram, type(_origin_histogram))
    # # 直方图均衡化
    # new_img = equalization_histogram(_origin_histogram, img)
    #
    # origin_grayHist = GrayHist(img)
    # equaliza_grayHist = GrayHist(new_img)
    # x = np.arange(256)
    # # 绘制灰度直方图
    # plt.figure(num=1)

    # plt.subplot(2, 2, 1)
    # plt.plot(x, origin_grayHist, 'r', linewidth=2, c='black')
    # plt.title("Origin")
    # plt.ylabel("number of pixels")
    # plt.subplot(2, 2, 2)
    # plt.plot(x, equaliza_grayHist, 'r', linewidth=2, c='black')
    # plt.title("Equalization")
    # plt.ylabel("number of pixels")
    # plt.subplot(2, 2, 3)
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.title('Origin')
    # plt.subplot(2, 2, 4)
    # plt.imshow(new_img, cmap=plt.cm.gray)
    # plt.title('Equalization')
    # plt.show()
