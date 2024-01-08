import numpy as np
import imageio
import cv2
import os.path
from sympy import symbols, solve


def xy_to_XYZ(xy: np.array, Y=1.0):
    X = xy[0] * Y / xy[1]
    Y = Y
    Z = (1 - (xy[0] + xy[1])) * Y / xy[1]

    return np.array([X, Y, Z])


def xyY_to_XYZ(xyY: np.array):
    X = xyY[0] * xyY[2] / xyY[1]
    Y = xyY[2]
    Z = (1 - (xyY[0] + xyY[1])) * xyY[2] / xyY[1]
    return np.array([X, Y, Z])


def P3_D65_to_XYZ(P3: np.array):
    P3_D65_to_XYZ_matrix = np.array([[0.486571, 0.265668, -0.198217],
                                     [0.228975, 0.691739, 0.079287],
                                     [0.000000, 0.045113, 1.043944]])

    XYZ = np.clip(np.dot(P3_D65_to_XYZ_matrix, P3), 0, None)

    return XYZ


def XYZ_to_P3_D65(XYZ: np.array):
    # XYZ_to_P3_D65_matrix = np.array([[2.493497, -0.931384, -0.402711],
    #                                  [-0.829489, 1.762664, 0.023625],
    #                                  [0.035846, -0.076172, 0.956885]])

    XYZ_to_P3_D65_matrix = np.array([[2.4934969, -0.9313836, -0.4027108],
                                     [-0.8294890, 1.7626641, 0.0236247],
                                     [0.0358458, -0.0761724, 0.9568845]])

    P3_D65_RGB = np.clip(np.dot(XYZ_to_P3_D65_matrix, XYZ), 0, None)

    return P3_D65_RGB

np_XYZ_to_P3_D65 = np.frompyfunc(XYZ_to_P3_D65, 1, 1)

def XYZ_to_xyY(XYZ: np.array):
    x = XYZ[0] / np.sum(XYZ)
    y = XYZ[1] / np.sum(XYZ)
    Y = XYZ[1]

    return np.array([x, y, Y])


def ST2084_dci_encode(a):
    c2 = 2413 * 32 / 4096
    c3 = 2392 * 32 / 4096
    c1 = c3 - c2 + 1
    k0 = 10000
    k1 = 4095
    m1 = 2610 / (4096 * 4)
    m2 = 2523 * 128 / 4096

    v1 = c1 + c2 * (pow(a / k0, m1))
    v2 = 1 + c3 * (pow(a / k0, m1))
    b = (1 / 2) + k1 * (pow(v1 / v2, m2))

    # return int(np.clip(b, 0, 4095))

    return int(np.floor(b)) / 4095


np_ST2084_dci_encode = np.frompyfunc(ST2084_dci_encode, 1, 1)


def Linear_to_ST2084(a):
    c2 = 2413 * 32 / 4096
    c3 = 2392 * 32 / 4096
    c1 = c3 - c2 + 1
    k0 = 10000.0
    m1 = 2610 / (4096 * 4)
    m2 = 2523 * 128 / 4096

    v1 = c1 + c2 * (pow(a / k0, m1))
    v2 = 1 + c3 * (pow(a / k0, m1))

    b = pow(v1 / v2, m2)

    return float(b)


np_linear_2084_encode = np.frompyfunc(Linear_to_ST2084, 1, 1)


def P3RGB_D65_ST2084_Chart_from_xy(xy, Y, width: int, height: int, url):
    XYZ_value = xy_to_XYZ(xy, Y)
    P3_RGB_value = XYZ_to_P3_D65(XYZ_value)

    image = np.zeros([height, width, 3], dtype=np.float32)

    image[0:height, 0:width] = [Linear_to_ST2084(P3_RGB_value[0]),
                                Linear_to_ST2084(P3_RGB_value[1]),
                                Linear_to_ST2084(P3_RGB_value[2])]

    imageio.imwrite(url, image)

    return image


def get_maxW_RGB_luminance(xy_W, xy_R, xy_G, xy_B, maxY_W):
    x_R = xy_R[0]
    y_R = xy_R[1]
    z_R = 1 - (x_R + y_R)

    x_G = xy_G[0]
    y_G = xy_G[1]
    z_G = 1 - (x_G + y_G)

    x_B = xy_B[0]
    y_B = xy_B[1]
    z_B = 1 - (x_B + y_B)

    XYZ_W = xy_to_XYZ(xy_W, maxY_W)

    matrix = np.array([[x_R / y_R, x_G / y_G, x_B / y_B],
                       [1, 1, 1],
                       [z_R / y_R, z_G / y_G, z_B / y_B]])

    maxY_RGB = np.linalg.solve(matrix, XYZ_W)

    return maxY_RGB


def get_maxT_XYZ_luminance(xy_W, xy_R, xy_G, xy_B, maxY_W, xy_T):
    x_R = xy_R[0]
    y_R = xy_R[1]
    z_R = 1 - (x_R + y_R)

    x_G = xy_G[0]
    y_G = xy_G[1]
    z_G = 1 - (x_G + y_G)

    x_B = xy_B[0]
    y_B = xy_B[1]
    z_B = 1 - (x_B + y_B)

    x_T = xy_T[0]
    y_T = xy_T[1]
    z_T = 1 - (x_T + y_T)

    XYZ_W = xy_to_XYZ(xy_W, maxY_W)

    matrix = np.array([[x_R / y_R, x_G / y_G, x_B / y_B],
                       [1, 1, 1],
                       [z_R / y_R, z_G / y_G, z_B / y_B]])

    maxY_RGB = np.linalg.solve(matrix, XYZ_W)

    XYZ_T_nominal = np.array([x_T / y_T, 1.0, z_T / y_T])
    nominal_RGB = np.linalg.solve(matrix, XYZ_T_nominal)
    np.seterr(divide='ignore', invalid='ignore')
    scalar = np.min(np.true_divide(maxY_RGB, nominal_RGB))
    XYZ_T_max = scalar * np.array(xy_to_XYZ(xy_T, 1.0))
    print(XYZ_T_max)
    return XYZ_T_max


# 生成H-K Effect的等亮度（Y值）色块

def HK_effect():
    # 色块xyY值
    c1_xy = [0.680, 0.320]
    c2_xy = [0.4, 0.4]
    c3_xy = [0.25, 0.6]
    c4_xy = [0.45, 0.3]
    c5_xy = [0.2, 0.3]
    c6_xy = [0.3, 0.15]
    c_list = [c1_xy, c2_xy, c3_xy, c4_xy, c5_xy, c6_xy]

    # get_maxW_RGB_luminance([0.3127, 0.3290], [0.680, 0.320], [0.265, 0.690], [0.150, 0.060], 1000)
    c1 = get_maxT_XYZ_luminance([0.3127, 0.3290], [0.680, 0.320], [0.265, 0.690], [0.150, 0.060], 1000, c1_xy)
    c2 = get_maxT_XYZ_luminance([0.3127, 0.3290], [0.680, 0.320], [0.265, 0.690], [0.150, 0.060], 1000, c2_xy)
    c3 = get_maxT_XYZ_luminance([0.3127, 0.3290], [0.680, 0.320], [0.265, 0.690], [0.150, 0.060], 1000, c3_xy)
    c4 = get_maxT_XYZ_luminance([0.3127, 0.3290], [0.680, 0.320], [0.265, 0.690], [0.150, 0.060], 1000, c4_xy)
    c5 = get_maxT_XYZ_luminance([0.3127, 0.3290], [0.680, 0.320], [0.265, 0.690], [0.150, 0.060], 1000, c5_xy)
    c6 = get_maxT_XYZ_luminance([0.3127, 0.3290], [0.680, 0.320], [0.265, 0.690], [0.150, 0.060], 1000, c6_xy)
    target_Y_luminance = 200.0

    W = 260
    H = 260
    W_margin = 130
    H_margin = 140
    w_dis = 20
    bg = 0.3

    i_W = 2 * W_margin + w_dis * (len(c_list) - 1) + len(c_list) * W
    i_H = 2 * H_margin + H

    deltaW = W + w_dis
    k = W_margin
    image_slide = np.zeros([i_H, i_W, 3], dtype=np.float32)#黑背景
    # image_slide = bg * np.ones([i_H, i_W, 3], dtype=np.float32)  # 灰背景
    for c in c_list:
        c_XYZ = xy_to_XYZ(c, target_Y_luminance)
        c_p3 = XYZ_to_P3_D65(c_XYZ)
        c_p3_2084 = np_linear_2084_encode(c_p3)
        # image_slide[H_margin:H_margin + H, k:k + W] = c_XYZ
        image_slide[H_margin:H_margin + H, k:k + W] = c_p3_2084
        k = k + deltaW

    imageio.imwrite(os.path.abspath('./p3_c1-c6_200nit_RGB_black_bg.exr'), image_slide)
    # imageio.imwrite(os.path.abspath('./XYZ_c1-c6_200nit_RGB_black_bg.exr'), image_slide)
    return


def Hunt_effect():
    Y_list_R = [0.3, 0.75, 1.5, 3, 6, 12, 24, 50, 100, 200]
    Y_list_G = [1.5, 3, 6, 12, 24, 36, 75, 150, 300, 600]
    Y_list_B = [0.1, 0.25, 0.5, 1, 2.25, 4.5, 9, 18, 36, 72]
    W = 160
    H = 160
    W_margin = 160
    H_margin = 40
    w_dis = 0
    bg = 0.3

    i_W = 2 * W_margin + w_dis * (len(Y_list_R) - 1) + len(Y_list_R) * W
    i_H = 2 * H_margin + H

    deltaW = W + w_dis
    k = W_margin
    image_r = np.zeros([i_H, i_W, 3], dtype=np.float32)  # 黑背景
    image_g = np.zeros([i_H, i_W, 3], dtype=np.float32)  # 黑背景
    image_b = np.zeros([i_H, i_W, 3], dtype=np.float32)  # 黑背景
    # image_slide = bg * np.ones([i_H, i_W, 3], dtype=np.float32)  # 灰背景
    for Y_R in Y_list_R:
        c_XYZ = xy_to_XYZ([0.680, 0.320], Y_R)
        c_p3 = XYZ_to_P3_D65(c_XYZ)
        c_p3_2084 = np_linear_2084_encode(c_p3)
        image_r[H_margin:H_margin + H, k:k + W] = c_p3_2084
        k = k + deltaW
    k = W_margin
    for Y_G in Y_list_G:
        c_XYZ = xy_to_XYZ([0.265, 0.690], Y_G)
        c_p3 = XYZ_to_P3_D65(c_XYZ)
        c_p3_2084 = np_linear_2084_encode(c_p3)
        image_g[H_margin:H_margin + H, k:k + W] = c_p3_2084
        k = k + deltaW
    k = W_margin
    for Y_B in Y_list_B:
        c_XYZ = xy_to_XYZ([0.150, 0.060], Y_B)
        c_p3 = XYZ_to_P3_D65(c_XYZ)
        c_p3_2084 = np_linear_2084_encode(c_p3)
        image_b[H_margin:H_margin + H, k:k + W] = c_p3_2084
        k = k + deltaW

    imageio.imwrite(os.path.abspath('./p3_red-10_RGB.exr'), image_r)
    imageio.imwrite(os.path.abspath('./p3_green-10_RGB.exr'), image_g)
    imageio.imwrite(os.path.abspath('./p3_blue-10_RGB.exr'), image_b)
    return


# def Stevens_effect():
#     Y_list_W = 1000.0 * np.ones([10])
#     for index, value in enumerate(Y_list_W):
#         Y_list_W[index] = pow(0.5, index) * Y_list_W[index]
#
#     Y_list_BG=np.linspace(0.0,200.0,1920,dtype=np.float32)
#     Y_list_BG=np_linear_2084_encode(Y_list_BG)
#     print(Y_list_BG)
#
#     W = 140
#     H = 140
#     W_margin = 80
#     H_margin = 90
#     w_dis = 40
#     bg = 0.3
#
#     i_W = 2 * W_margin + w_dis * (len(Y_list_W) - 1) + len(Y_list_W) * W
#     i_H = 2 * H_margin + H
#     k = W_margin
#     deltaW = W + w_dis
#
#     image_BG_grey=np.tile(Y_list_BG.reshape(1,1920),(320,1))
#     image_BG_grey=np.expand_dims(image_BG_grey,axis=2)
#     image_BG=np.array(np.repeat(image_BG_grey,3,axis=2),dtype=np.float32)
#     print(image_BG.dtype)
#
#     # image= image_BG
#     image= np.ones([i_H, i_W,3], dtype=np.float32)
#     print(image.dtype)
#
#     for Y_W in Y_list_W:
#         c_XYZ = xy_to_XYZ([0.3127, 0.3290], Y_W)
#         c_p3 = XYZ_to_P3_D65(c_XYZ)
#         c_p3_2084 = np_linear_2084_encode(c_p3)
#         image[H_margin:H_margin + H, k:k + W] = c_p3_2084
#         k = k + deltaW
#     imageio.imwrite(os.path.abspath('./steve.exr'), image)
#     return




if __name__ == '__main__':
    # P3RGB_D65_ST2084_Chart_from_xy(c1_xy,200.0,128,128,os.path.abspath('./p3_c1_200nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy(c2_xy,200.0,128,128,os.path.abspath('./p3_c2_200nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy(c3_xy,200.0,128,128,os.path.abspath('./p3_c3_200nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy(c4_xy,200.0,128,128,os.path.abspath('./p3_c4_200nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy(c5_xy,200.0,128,128,os.path.abspath('./p3_c5_200nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy(c6_xy,200.0,128,128,os.path.abspath('./p3_c6_200nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy([0.3127,0.3290],0.05,1920,1080,os.path.abspath('./p3_white_0.05nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy([0.3127,0.3290],1.0,1920,1080,os.path.abspath('./p3_white_1nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy([0.3127,0.3290],5.0,1920,1080,os.path.abspath('./p3_white_5nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy([0.3127,0.3290],10.0,1920,1080,os.path.abspath('./p3_white_10nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy([0.3127,0.3290],40.0,1920,1080,os.path.abspath('./p3_white_40nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy([0.3127,0.3290],100.0,1920,1080,os.path.abspath('./p3_white_100nit_RGB.exr'))
    # P3RGB_D65_ST2084_Chart_from_xy([0.3127,0.3290],200.0,1920,1080,os.path.abspath('./p3_white_200nit_RGB.exr'))

    HK_effect()
    # Hunt_effect()
    # Stevens_effect()
