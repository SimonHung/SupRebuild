import numpy as np

# https://mymusing.co/bt-709-yuv-to-rgb-conversion-color/
#
# Computer RGB To YCbCr

matrix = np.array([[46.742, 157.243, 15.874],
                   [-25.765, -86.674, 112.439],
                   [112.439, -102.129, -10.310]])


matrix /= 256
print("[Y, Cb, Cr] = [16, 128, 128] + [3X3 matrix] * [R, G, B]")
print("\n[3X3 matrix]:")
print(np.around(matrix, decimals=3))
print("")

print("[R, G, B] = [3X3 inverse Matrix] * [Y-16, Cb-128, Cr-128]")
print("\n[3X3 inverse matrix]:")
inverse_matrix = np.linalg.inv(matrix)
print(np.around(inverse_matrix, decimals=3))
