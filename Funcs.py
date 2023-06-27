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


def median_filter_adapt(image, kernel_size):
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
            temp_std = np.std(window)
            print(temp_std)
            filtered_image[i, j] = np.median(window, axis=(0, 1))

    return filtered_image


if __name__ == '__main__':
    img = cv2.imread("imgs/b.png")
    median_filter_adapt(img, (5, 5))
    pass
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
