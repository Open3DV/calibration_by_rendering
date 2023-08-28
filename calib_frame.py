"""
    Stereo calibrate
"""

import os
import cv2
import numpy as np

# from utils import *

image_nums = 45

params = cv2.SimpleBlobDetector_Params()
params.maxArea = 10e4
params.minArea = 8
params.minDistBetweenBlobs = 4
blobDetector = cv2.SimpleBlobDetector_create(params)


def single_calibrate(path, is_left):
    image_pts_list = []
    obj_pts_list = []

    obj_pts = []

    image_list = []
    # for row in range(pattern_size[1]):
    #     for col in range(pattern_size[0]):
    #         obj_pts.append([row * resolution, col * resolution, 0])

    for row in range(pattern_size[1]):
        for col in range(pattern_size[0]):
            if row % 2 == 0:
                obj_pts.append([row * resolution * 0.5, col * resolution, 0])
            else:
                obj_pts.append([row * resolution * 0.5, col * resolution + 0.5 * resolution, 0])

    for i in range(image_nums):
        if i < 10:
            # img_path = path + "/0" + str(i) + ("/phase12.bmp" if is_left else "/phase26.bmp")
            img_path = path + "/0" + str(i) + ("-L.bmp" if is_left else "-R.bmp")
        else:
            # img_path = path + "/" + str(i) + ("/phase12.bmp" if is_left else "/phase26.bmp")
            img_path = path + "/" + str(i) + ("-L.bmp" if is_left else "-R.bmp")
        print('img: ', img_path)
        img = cv2.imread(img_path, 0)
        # cv2.equalizeHist(img, img)
        img = cv2.bitwise_not(img)

        # is_found, corners_nx1x2 = cv2.findChessboardCorners(img, patternSize=pattern_size)
        # 对图片取负片
        # for row in range(img.shape[0]):
        #     for col in range(img.shape[1]):
        #         img[row][col] = 255 - img[row][col]

        is_found, corners_nx1x2 = cv2.findCirclesGrid(img, patternSize=pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if not is_found:
            print("not find!")
            continue
        cv2.drawChessboardCorners(img, pattern_size, corners_nx1x2, is_found)
        cv2.namedWindow('img', 0)
        cv2.imshow('img', img)
        cv2.waitKey(30)

        if is_found:
            image_pts_list.append(np.float32(corners_nx1x2).reshape(-1, 2))
            obj_pts_list.append(np.float32(obj_pts))
            image_list.append(i)

    (
        re_projection_err,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        _,
        _,
        re_projection_errs,
    ) = cv2.calibrateCameraExtended(
        objectPoints=obj_pts_list,
        imagePoints=image_pts_list,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        # flags=cv2.CALIB_RATIONAL_MODEL
    )

    print('err: ', re_projection_err)
    return camera_matrix, dist_coeffs, image_pts_list, obj_pts_list, image_list


def single_calibrate_use_point(path, is_left):
    image_pts_list = []
    obj_pts_list = []

    obj_pts = []

    image_list = []
    # for row in range(pattern_size[1]):
    #     for col in range(pattern_size[0]):
    #         obj_pts.append([row * resolution, col * resolution, 0])

    for row in range(pattern_size[1]):
        for col in range(pattern_size[0]):
            if row % 2 == 0:
                obj_pts.append([row * resolution * 0.5, col * resolution, 0])
            else:
                obj_pts.append([row * resolution * 0.5, col * resolution + 0.5 * resolution, 0])

    for i in range(image_nums):
        if i < 10:
            # img_path = path + "/0" + str(i) + ("/phase12.bmp" if is_left else "/phase26.bmp")
            img_path = path + "/0" + str(i) + ("_L.xml" if is_left else "_R.xml")
        else:
            # img_path = path + "/" + str(i) + ("/phase12.bmp" if is_left else "/phase26.bmp")
            img_path = path + "/" + str(i) + ("_L.xml" if is_left else "_R.xml")
        print('img: ', img_path)
        img = cv2.imread(img_path, 0)
        # cv2.equalizeHist(img, img)
        img = cv2.bitwise_not(img)

        is_found = True
        corners_nx1x2 = read_points_from_file(img_path)
        if not is_found:
            print("not find!")
            continue
        cv2.drawChessboardCorners(img, pattern_size, corners_nx1x2, is_found)
        cv2.namedWindow('img', 0)
        cv2.imshow('img', img)
        cv2.waitKey(30)

        if is_found:
            image_pts_list.append(np.float32(corners_nx1x2).reshape(-1, 2))
            obj_pts_list.append(np.float32(obj_pts))
            image_list.append(i)

    (
        re_projection_err,
        camera_matrix,
        dist_coeffs,
        rvecs,
        tvecs,
        _,
        _,
        re_projection_errs,
    ) = cv2.calibrateCameraExtended(
        objectPoints=obj_pts_list,
        imagePoints=image_pts_list,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        # flags=cv2.CALIB_RATIONAL_MODEL
    )

    print('err: ', re_projection_err)
    return camera_matrix, dist_coeffs, image_pts_list, obj_pts_list, image_list


def read_points_from_file(file_path):
    input_xml = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    input_mat = input_xml.getNode('cam_points').mat()
    return input_mat


img_l_path = "./board_points"
img_r_path = "./board_points"

pattern_size = (7, 11)
resolution = 40
image_size = (1624, 1240)

camera_matrix_l, dist_coeffs_l, image_pts_list_l, obj_pts_list_l, image_list_num_l = single_calibrate_use_point(
    img_l_path, True)
print("camera_matrix_l", camera_matrix_l)
print("dist_coeffs_l", dist_coeffs_l)
camera_matrix_r, dist_coeffs_r, image_pts_list_r, obj_pts_list_r, image_list_num_r = single_calibrate_use_point(
    img_r_path, False)
print("camera_matrix_r", camera_matrix_r)
print("dist_coeffs_r", dist_coeffs_r)

# cv_file = cv2.FileStorage("calib_param.xml", cv2.FILE_STORAGE_WRITE)
# cv_file.write("camera_intrinsic_l", camera_matrix_l)
# cv_file.write("camera_dist_l", dist_coeffs_l)
# cv_file.write("camera_intrinsic_r", camera_matrix_r)
# cv_file.write("camera_dist_r", dist_coeffs_r)

image_pts_list_l_temp = []
image_pts_list_r_temp = []
obj_pts_list_l_temp = []
obj_pts_list_r_temp = []

for i in range(image_nums):
    if i in image_list_num_l and i in image_list_num_r:
        for j in range(len(image_list_num_l)):
            if i == image_list_num_l[j]:
                image_pts_list_l_temp.append(image_pts_list_l[j])
                obj_pts_list_l_temp.append(obj_pts_list_l[j])
                break
        for j in range(len(image_list_num_r)):
            if i == image_list_num_r[j]:
                image_pts_list_r_temp.append(image_pts_list_r[j])
                obj_pts_list_r_temp.append(obj_pts_list_r[j])
                break

image_pts_list_l = image_pts_list_l_temp
image_pts_list_r = image_pts_list_r_temp
obj_pts_list_l = obj_pts_list_l_temp
obj_pts_list_r = obj_pts_list_r_temp

# camera_matrix_3d_cam = np.array([[2404.02,      0,      963.93],
# [0,     2404.37,        569.764],
# [0,     0,      1]
# ], dtype=np.float64)
#
# dist_coeff_3d_cam = np.array([-0.0913767,   0.129795,   -0.000946345,   0.000416301,    -0.0101731], dtype=np.float64)

assert len(image_pts_list_l) == len(image_pts_list_r), "not found equal num of boards"

(
    re_projection_err,
    camera_matrix_1,
    dist_coeffs_1,
    camera_matrix_2,
    dist_coeffs_2,
    R,
    T,
    E,
    F,
    re_projection_errs,
) = cv2.stereoCalibrateExtended(
    objectPoints=obj_pts_list_l,
    imagePoints1=image_pts_list_l,
    imagePoints2=image_pts_list_r,
    imageSize=image_size,
    cameraMatrix1=camera_matrix_l,
    distCoeffs1=dist_coeffs_l,
    cameraMatrix2=camera_matrix_r,
    distCoeffs2=dist_coeffs_r,
    flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    # flags=cv2.CALIB_FIX_INTRINSIC,
    R=None,
    T=None,
)

print('stereo err: ', re_projection_err)
print('camera_matrix_1: ', camera_matrix_1)
print('dist_coeffs_1: ', dist_coeffs_1)

print('camera_matrix_2: ', camera_matrix_2)
print('dist_coeffs_2: ', dist_coeffs_2)
print('R: ', R)
print('T: ', T)

cv_file = cv2.FileStorage("calib_param.xml", cv2.FILE_STORAGE_WRITE)
cv_file.write("camera_intrinsic_l", camera_matrix_l)
cv_file.write("camera_dist_l", dist_coeffs_l)
cv_file.write("camera_intrinsic_r", camera_matrix_r)
cv_file.write("camera_dist_r", dist_coeffs_r)
cv_file.write("R", R)
cv_file.write("T", T)

cv_file.release()
print('save xml success!')