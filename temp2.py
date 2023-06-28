import math

import numpy as np


# with open("a.txt", "rb") as f:
#     print(f.read(4))

# bit_str = "11110001"  # 8位的0和1字符串
# byte_val = int(bit_str, 2)  # 将二进制字符串转换为整数
#
# with open("output.bin", "wb") as file:  # 在二进制模式下打开文件
#     file.write(bytearray([byte_val]))  # 将字节写入文件

# with open("output.bin", "rb") as f:
#     a = f.read(1)
#
# byte_str = bin(int.from_bytes(a, "big"))[2:].zfill(8)
# print(byte_str)

#
# def write_bytes_img(array: np.ndarray):
#     pass


# with open("output.bin", "rb") as file:  # 以二进制模式打开文件
#     byte_data = file.read()  # 读取整个文件为字节数据
#     msg = ""
#
#     bit_sequence = ""  # 存储0和1的序列
#     for byte in byte_data:
#         byte_str = bin(byte)[2:].zfill(8)  # 将每个字节转换为二进制字符串
#
#         for bit in byte_str:
#             bit_sequence += bit  # 将每个位添加到序列中
#
#             if len(bit_sequence) == 8:
#                 # 到达8位时，恢复一个0和1序列
#                 # print(bit_sequence)
#                 msg += bit_sequence
#                 bit_sequence = ""  # 重新开始一个新的序列
#
#     if bit_sequence:  # 处理最后的不足8位的序列
#         print(bit_sequence, "eeeee")
#
# with open("1.txt", "w", encoding="utf-8") as f:
#     f.write(msg)
#
# print(msg)


# with open("1.txt", "r") as f:
#     a = f.read()
#
# with open("2.txt", "r") as f:
#     b = f.read()
#
# print(a == b)


# with open("output.bin", "rb") as file:  # 以二进制模式打开文件
#     byte_data = file.read(5)  # 读取整个文件为字节数据
#
#     print(bin(byte_data[0])[2:].zfill(8), end="")
#     print(bin(byte_data[1])[2:].zfill(8))
#
#     print(bin(byte_data[2])[2:].zfill(8), end="")
#     print(bin(byte_data[3])[2:].zfill(8))
#
#     print(int(bin(byte_data[4])[2:].zfill(8), 2))
#
#     byte_data = file.read(1)  # 读取整个文件为字节数据
#     print(byte_data)
#     print(bin(byte_data[0])[2:].zfill(8), end="")


# with open("output.bin", "rb") as file:  # 以二进制模式打开文件
#     byte_data = file.read(5)  # 读取整个文件为字节数据
#     msg = ""
#
#     bit_sequence = ""  # 存储0和1的序列
#     for byte in byte_data:
#         byte_str = bin(byte)[2:].zfill(8)  # 将每个字节转换为二进制字符串
#
#         for bit in byte_str:
#             bit_sequence += bit  # 将每个位添加到序列中
#
#             if len(bit_sequence) == 8:
#                 # 到达8位时，恢复一个0和1序列
#                 print(bit_sequence)
#                 msg += bit_sequence
#                 bit_sequence = ""  # 重新开始一个新的序列


def read_bytes():
    with open("output.bin", "rb") as file:
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


