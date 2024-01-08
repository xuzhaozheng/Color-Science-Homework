from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def grey_world_white_balance(image_array):
    # 计算每个通道的平均值
    avg_r = np.mean(image_array[:, :, 0])
    avg_g = np.mean(image_array[:, :, 1])
    avg_b = np.mean(image_array[:, :, 2])

    # 计算白平衡系数
    gray_world_coeff = [avg_g / avg_r, 1.0, avg_g / avg_b]

    # 应用白平衡系数
    balanced_image_array = image_array * gray_world_coeff
    balanced_image_array = np.clip(balanced_image_array, 0, 255)

    return balanced_image_array


def white_patch_white_balance(image_array):
    # 计算每个通道的最大值
    max_r = np.max(image_array[:, :, 0])
    max_g = np.max(image_array[:, :, 1])
    max_b = np.max(image_array[:, :, 2])

    # 计算白平衡系数
    white_patch_coeff = [max_g / max_r, 1.0, max_g / max_b]

    # 应用白平衡系数
    balanced_image_array = image_array * white_patch_coeff
    balanced_image_array = np.clip(balanced_image_array, 0, 255)

    return balanced_image_array


def K(x, threshold_a, threshold_b):
    if np.abs(x) >= threshold_a:
        out = 2 * np.sign(x)
    elif threshold_b <= np.abs(x) < threshold_a:
        out = np.sign(x)
    elif 0 <= np.abs(x) < threshold_b:
        out = 0

    return out

def rgb_to_yuv(rgb_image):
    rgb_yuv_matrix = np.array([[0.299, 0.587, 0.114],
                               [-0.299, -0.587, 0.886],
                               [0.701, -0.587, -0.114]])
    yuv_image = np.dot(rgb_image.astype(np.float32), rgb_yuv_matrix.T)

    return yuv_image

def iterative_white_balance(image_array, max_iterations=15):
    current_rgb_array = image_array
    T = 0.5

    for iteration in range(max_iterations):
        # 转化为YUV
        yuv_array = rgb_to_yuv(current_rgb_array)
        # 提取灰点,实际可能没有灰世界假设法好用
        condition = ((np.abs(yuv_array[:, :, 1]) + np.abs(yuv_array[:, :, 2])) / (yuv_array[:, :, 0] + 1e-8)) < T
        filtered_grey_point = yuv_array[condition]

        avg_u = np.mean(filtered_grey_point[:, 1])
        avg_v = np.mean(filtered_grey_point[:, 2])

        # print(filtered_grey_point.shape)
        # print(avg_u)
        # print(avg_v)

        abs_avg_u = np.abs(avg_u)
        abs_avg_v = np.abs(avg_v)

        phi = 0.0
        w_r = 1.0
        w_g = 1.0
        w_b = 1.0

        a = 0.8
        b = 0.01
        mu = 0.0312
        d = 0
        if abs_avg_u > abs_avg_v or (abs_avg_u == abs_avg_v and abs_avg_u != 0):
            phi = avg_u
            varepsilon = d - phi
            w_b = w_b + mu * K(varepsilon, a, b)

        elif abs_avg_u < abs_avg_v:
            phi = avg_v
            varepsilon = d - phi
            w_r = w_r + mu * K(varepsilon, a, b)
        elif abs_avg_u == abs_avg_v == 0.0:
            phi = 0.0
            varepsilon = d - phi

        if varepsilon == 0:
            break;

        iterative_white_coeff = [w_r, w_g, w_b]
        # print(iterative_white_coeff)
        # print('')
        current_rgb_array = current_rgb_array * iterative_white_coeff
        current_rgb_array = np.clip(current_rgb_array, 0, 255)

    balanced_image_array = current_rgb_array

    return balanced_image_array


def oetf_linear_to_srgb_int8(o):
    o_normalized = o / 255.0
    e_normalized = np.where(o_normalized <= 0.0031308, 12.92 * o_normalized,
                            1.055 * (o_normalized ** (1.0 / 2.4)) - 0.055)

    e = 255 * e_normalized
    return e


# np_oetf_linear_to_srgb_int8 = np.frompyfunc(oetf_linear_to_srgb_int8, 1, 1)


# 读取图像
image_path = "Test01_00090000.tif"
image = Image.open(image_path)
image_array = np.array(image)

# 灰色世界假设法白平衡处理
grey_world_balanced_array = grey_world_white_balance(image_array)

# 白斑法白平衡处理
white_patch_balanced_array = white_patch_white_balance(image_array)

# 迭代白平衡法处理
iterative_balanced_array = iterative_white_balance(image_array)

# 添加Gamma校正
original_gamma_corrected_array = oetf_linear_to_srgb_int8(image_array)
grey_world_gamma_corrected_array = oetf_linear_to_srgb_int8(grey_world_balanced_array)
white_patch_gamma_corrected_array = oetf_linear_to_srgb_int8(white_patch_balanced_array)
iterative_gamma_corrected_array = oetf_linear_to_srgb_int8(iterative_balanced_array)

# 创建子图
fig, axs = plt.subplots(2, 3, figsize=(10, 5))

# 原图像
axs[0, 0].imshow(image)
axs[0, 0].set_title("Original Linear Image")

original_gamma_corrected_image = Image.fromarray(original_gamma_corrected_array.astype(np.uint8))
axs[0, 1].imshow(original_gamma_corrected_image)
axs[0, 1].set_title("Original Image")

# 灰色世界假设法白平衡后的图像
grey_world_balanced_image = Image.fromarray(grey_world_gamma_corrected_array.astype(np.uint8))
axs[1, 0].imshow(grey_world_balanced_image)
axs[1, 0].set_title("Grey World White Balanced Image")

# 白斑法白平衡后的图像
white_patch_balanced_image = Image.fromarray(white_patch_gamma_corrected_array.astype(np.uint8))
axs[1, 1].imshow(white_patch_balanced_image)
axs[1, 1].set_title("White Patch White Balanced Image")

# 迭代白平衡法白平衡后的图像
iterative_balanced_image = Image.fromarray(iterative_gamma_corrected_array.astype(np.uint8))
axs[1, 2].imshow(iterative_balanced_image)
axs[1, 2].set_title("Iterative White Balanced Image")

# 去除坐标轴
for ax in axs.flat:
    ax.axis("off")

# 调整子图间的距离
plt.tight_layout()

# 显示图像
plt.show()
