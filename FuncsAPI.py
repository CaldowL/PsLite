import time
from matplotlib import pyplot as plt

import Utils
import model
from Funcs import *
from main import Form


def api_histogram_equalization(window, showDetail=False, color=False):
    if not color:
        img = cv2.imread(model.file_select_path, cv2.IMREAD_GRAYSCALE)
        _origin_histogram = origin_histogram(img)
        new_img = equalization_histogram(_origin_histogram, img)
        origin_grayHist = GrayHist(img)
        equaliza_grayHist = GrayHist(new_img)
    else:
        img = cv2.imread(model.file_select_path, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        _origin_histogram = origin_histogram(img[:, :, 2])
        new_img = equalization_histogram(_origin_histogram, img[:, :, 2])
        origin_grayHist = GrayHist(img[:, :, 2])
        equaliza_grayHist = GrayHist(new_img)

    if showDetail:
        x = np.arange(256)
        # 绘制灰度直方图
        plt.figure(num=1)

        plt.plot(x, origin_grayHist, 'r', linewidth=2, c='black')
        plt.title("Origin")
        plt.show()

        plt.plot(x, equaliza_grayHist, 'r', linewidth=2, c='black')
        plt.title("Equalization")
        plt.show()

    if not color:
        window.label_4.setPixmap(Utils.img_to_pixmap(new_img))
        cv2.imwrite(f"results/{time.time()}_gray.jpg", new_img)
        window.set_status("灰度图像直方图均衡完成")
    else:
        new_img_show = np.zeros(shape=(new_img.shape[0], new_img.shape[1], 3))
        new_img_show[:, :, 0] = img[:, :, 0]
        new_img_show[:, :, 1] = img[:, :, 1]
        new_img_show[:, :, 2] = new_img[:, :]

        new_img_show = cv2.cvtColor(new_img_show.astype(np.uint8), cv2.COLOR_HSV2BGR)

        window.label_4.setPixmap(Utils.img_to_pixmap(new_img_show))
        window.set_status("彩色图像直方图均衡完成")
        cv2.imwrite(f"results/{time.time()}_color.jpg", new_img_show)


def api_median_filter(window, kernel_size=(3, 3), kernel_type="矩形框"):
    assert model.file_select_ndarray is not None, "请先选择图片"
    assert kernel_type in ["矩形框", "圆形框", "菱形框"], "暂不支持此类型"
    window.set_status(f"开始图像中值滤波  滤波类型：{kernel_type}")
    t = time.time()
    img = model.file_select_ndarray
    if kernel_type == "矩形框":
        img = median_filter_square(img, kernel_size)
    elif kernel_type == "圆形框":
        img = median_filter_circle(img, kernel_size[0])
    elif kernel_type == "菱形框":
        img = median_filter_diamond(img, kernel_size[0])

    window.label_4.setPixmap(Utils.img_to_pixmap(img))
    window.set_status(f"图像中值滤波完成,耗时: {round(time.time() - t, 2)}秒")
