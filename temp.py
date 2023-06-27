import numpy as np

# 创建一个大小为 (7, 7) 的二维布尔类型数组，作为示例
arr = np.array([[False, False, False, True, True, True, True],
                [False, False, True, True, True, True, True],
                [False, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True]], dtype=bool)

# print(np.logical_xor(np.flip(arr), arr))
# print(np.flip(np.logical_xor(np.flip(arr), arr), axis=1))
print(np.logical_not(np.logical_xor(np.flip(np.logical_xor(np.flip(arr), arr), axis=1), np.logical_xor(np.flip(arr), arr))))
