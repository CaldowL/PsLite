import numpy as np


def image_encode_square(image: np.ndarray, kernel_r: int):
    image = image[:, :, 0]
    rows, cols = image.shape[:2]
    rows -= rows % kernel_r
    cols -= cols % kernel_r

    filtered_image = np.zeros((rows, cols))
    for i in range(0, rows, kernel_r):
        for j in range(0, cols, kernel_r):
            window = image[i:i + kernel_r, j: j + kernel_r]
            ave = int(np.average(window))
            # print(ave)

            for m in range(kernel_r):
                for n in range(kernel_r):
                    index_x = i + m
                    index_y = j + n

                    filtered_image[index_x][index_y] = window[m][n] >= ave  # k * k

    t_img = filtered_image.flatten()
    t_img.dtype = np.bool

    return filtered_image * 255
