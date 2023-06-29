import time

import win32con
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
    assert kernel_type in ["矩形框", "圆形框", "菱形框", "自适应滤波"], "暂不支持此类型"
    window.set_status(f"开始图像中值滤波  滤波类型：{kernel_type}")
    t = time.time()
    img = model.file_select_ndarray
    if kernel_type == "矩形框":
        img = median_filter_square(img, kernel_size)
    elif kernel_type == "圆形框":
        img = median_filter_circle(img, kernel_size[0])
    elif kernel_type == "菱形框":
        img = median_filter_diamond(img, kernel_size[0])
    elif kernel_type == "自适应滤波":
        img = median_filter_adapt(img, 3, 25)

    window.label_4.setPixmap(Utils.img_to_pixmap(img))
    window.set_status(f"图像中值滤波完成,耗时: {round(time.time() - t, 2)}秒")


def api_square_encode(window, kernel_size):
    assert model.file_select_ndarray is not None, "请先选择图片"
    window.set_status(f"开始图像方块压缩编码")
    t = time.time()
    file_name = f"results/{int(t)}.dat"
    img = model.file_select_ndarray
    image_encode_square(img, kernel_size, file_name)
    window.set_status(f"图像方块压缩编码完成,耗时: {round(time.time() - t, 2)}秒")

    size_ori = os.path.getsize(model.file_select_path)
    size_zip = os.path.getsize(file_name)
    msg = "方块编码完成\n"
    msg += f"压缩文件位置: {file_name}\n"
    msg += f"原文件大小: {size_ori} Bytes\n"
    msg += f"压缩后文件大小: {size_zip} Bytes\n"
    msg += f"文件压缩比: {round(size_ori / size_zip, 2)}\n"
    msg += f"文件压缩率: {round(size_zip * 100 / size_ori, 2)} %\n"
    msg += "点击确认打开文件夹\n"

    if Utils.show_information(msg, True) == 1:
        os.system(rf"explorer /select,{os.path.abspath(file_name)}")


def api_square_decode(window, file_path):
    window.set_status(f"开始图像方块压缩解码")
    t = time.time()
    file_name = f"results/{int(t)}.bmp"
    image_decode_square(file_path, file_name)

    window.set_status(f"图像方块压缩解码完成,耗时: {round(time.time() - t, 2)}秒")

    img = cv2.imread(file_name)
    model.file_select_ndarray = img
    window.label_4.setScaledContents(True)
    window.label_4.setPixmap(Utils.img_to_pixmap(img))

    msg = "方块解码完成\n"
    msg += f"解压后文件位置: {file_name}\n"
    msg += "点击确认打开文件夹\n"

    if Utils.show_information(msg, True) == 1:
        os.system(rf"explorer /select,{os.path.abspath(file_name)}")
