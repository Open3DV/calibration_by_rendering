# 计算findcircleGrid()提取的圆心到渲染图像真实圆心的距离

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

errs_list_l = []
errs_list_r = []
num_list_l = []
num_list_r = []


def read_points_from_file(file_path):
    input_xml = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    input_mat = input_xml.getNode('cam_points').mat()
    return input_mat


def calculate_distance(point_path, img_path, is_left):
    # 将图片数据分为左右目，并读取真实点坐标
    for idx in range(pic_num):
        if is_left:
            image_path = img_path + "\\" + "%02d_L.png" % idx
            points_path = point_path + "\\" + "%02d_L.xml" % idx
            point_mat_l = read_points_from_file(points_path)
            point_mat_l = point_mat_l.reshape(-1, 2)
        else:
            image_path = img_path + "\\" + "%02d_R.png" % idx
            points_path = point_path + "\\" + "%02d_R.xml" % idx
            point_mat_r = read_points_from_file(points_path)
            point_mat_r = point_mat_r.reshape(-1, 2)

        img = cv2.imread(image_path)
        img = cv2.bitwise_not(img)

        # 提取渲染标定板图片的图像点坐标
        is_found, corners = cv2.findCirclesGrid(img, patternSize=pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if not is_found:
            print(image_path + "：未找到圆心")
            continue
        # 作差
        if is_found:
            if is_left:
                corners_L = corners.reshape(-1, 2)
                errs_l = corners_L - point_mat_l
                errs_list_l.append(errs_l)
                num_list_l.append(idx)
                print(errs_l)
            else:
                corners_R = corners.reshape(-1, 2)
                errs_r = corners_R - point_mat_r
                errs_list_r.append(errs_r)
                num_list_r.append(idx)
                print(errs_r)

        # 根据图像点与真实点的差值绘制表征偏移量方向和大小的直线
        if is_left:
            for num in range(len(errs_l)):
                corner_L = tuple(np.int32(corners_L[num]))
                offset_L = tuple(np.int32(point_mat_l[num]) + np.int32(np.ones(2) * 500 * errs_l[num]))
                cv2.line(img, corner_L, offset_L, (0, 255, 0), 2)
        else:
            for num in range(len(errs_r)):
                corner_R = tuple(np.int32(corners_R[num]))
                offset_R = tuple(np.int32(point_mat_r[num]) + np.int32(np.ones(2) * 500 * errs_r[num]))
                cv2.line(img, corner_R, offset_R, (0, 255, 0), 2)

        cv2.imshow(image_path, img)
        cv2.waitKey(0)


pic_num = 45
pattern_size = (7, 11)
point_path = "./board_points"
img_path = "./render_result"
calculate_distance(img_path=img_path, point_path=point_path, is_left=True)
calculate_distance(img_path=img_path, point_path=point_path, is_left=False)

# 左目相机图像点与真实点分布散点图
# 取每张标定板的平均距离差来表征整体差距
average_errs_l = []
average_errs_r = []
errs_list_l = np.array(errs_list_l)
for i in range(len(errs_list_l)):
    average_err = np.mean(errs_list_l[i], axis=0)
    average_errs_l.append(average_err)

average_errs_l = np.array(average_errs_l)
print(average_errs_l)
x_l = average_errs_l[:, 0]
y_l = average_errs_l[:, 1]
fig, ax = plt.subplots()
ax.scatter(x_l, y_l, c='r', label='$camera1$')
for i in range(len(x_l)):
    ax.text(x_l[i], y_l[i], num_list_l[i])

# 右目相机图像点与真实点分布散点图
for i in range(len(errs_list_r)):
    average_err = np.mean(errs_list_r[i], axis=0)
    average_errs_r.append(average_err)

average_errs_r = np.array(average_errs_r)
print(average_errs_r)
x_r = average_errs_r[:, 0]
y_r = average_errs_r[:, 1]
ax.scatter(x_r, y_r, c='b', label='$camera2$')
for i in range(len(x_l)):
    ax.text(x_r[i], y_r[i], num_list_r[i])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Center Distance')
plt.legend()
plt.show()
