import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

num = 45
pattern_size = (7, 11)
image_size = (1624, 1240)
imgs_path = "./render_result/"
save_path = "./undistorted_point/"
point_path = "./board_points"

fs1 = cv2.FileStorage("calib_param.xml", cv2.FileStorage_READ)
camera_matrix1_real = fs1.getNode("camera_intrinsic_l").mat()
dsitcoeffs1_real = fs1.getNode("camera_dist_l").mat()
camera_matrix2_real = fs1.getNode("camera_intrinsic_r").mat()
dsitcoeffs2_real = fs1.getNode("camera_dist_r").mat()
R_real = fs1.getNode("R").mat()
T_real = fs1.getNode("T").mat()
fs1.release()

# 根据标定图片的张数将fs2中的文件名进行修改
fs2 = cv2.FileStorage("calib_param_45.xml", cv2.FileStorage_READ)
camera_matrix1 = fs2.getNode("camera_intrinsic_l").mat()
dsitcoeffs1 = fs2.getNode("camera_dist_l").mat()
camera_matrix2 = fs2.getNode("camera_intrinsic_r").mat()
dsitcoeffs2 = fs2.getNode("camera_dist_r").mat()
R = fs2.getNode("R").mat()
T = fs2.getNode("T").mat()
fs2.release()


# 读图像点
def read_points_from_file(file_path):
    input_xml = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    input_mat = input_xml.getNode('cam_points').mat()
    return input_mat


# 提取图像点
def abs_point_from_pic(path, is_left):
    for i in range(num):
        if is_left:
            img_path = path + "%02d_L.png" % i
        else:
            img_path = path + "%02d_R.png" % i

        img = cv2.imread(img_path)
        img = cv2.bitwise_not(img)
        is_found, corners = cv2.findCirclesGrid(img, patternSize=pattern_size, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if not is_found:
            print(img_path + ":未找到圆心")
            continue
        else:
            if is_left:
                cv_file = cv2.FileStorage(save_path + "%02d_L.xml" % i, cv2.FileStorage_WRITE)
                cv_file.write("cam_points", corners)
                cv_file.release()
            else:
                cv_file = cv2.FileStorage(save_path + "%02d_R.xml" % i, cv2.FileStorage_WRITE)
                cv_file.write("cam_points", corners)
                cv_file.release()


# 图像点去畸变
def undistorted_point(point, is_left):
    if is_left:
        undis_point = cv2.undistortPoints(point, cameraMatrix=camera_matrix1, distCoeffs=dsitcoeffs1, P=camera_matrix1)
    else:
        undis_point = cv2.undistortPoints(point, cameraMatrix=camera_matrix2, distCoeffs=dsitcoeffs2, P=camera_matrix2)
    return undis_point


# 计算距离
def distance(x, y):
    dis = np.sqrt((y[0] - x[0]) ** 2 + (y[1] - x[1]) ** 2 + (y[2] - x[2]) ** 2)
    return dis


abs_point_from_pic(imgs_path, is_left=True)
abs_point_from_pic(imgs_path, is_left=False)


Rl = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]], dtype=np.float64)
# Rl_rod = cv2.Rodrigues(Rl)
tl = np.array([[0], [0], [0]], dtype=np.float64)
P1 = np.dot(camera_matrix1, np.hstack((Rl, tl)))
P2 = np.dot(camera_matrix2, np.hstack((R, T)))
# print("camera1投影矩阵：", P1)
# print("camera2投影矩阵：", P2)


# 设置一个自定义的空间点
point1 = np.array([300, 300, 1500], dtype=np.float64)
point2 = np.array([500, 500, 1400], dtype=np.float64)

# 左目第1个点对应的像素坐标
point_l1, _ = cv2.projectPoints(point1, Rl, tl, cameraMatrix=camera_matrix1_real, distCoeffs=dsitcoeffs1_real)
point_l1 = undistorted_point(point_l1, True)
# print(point1_l)

# 左目第2个点对应的像素坐标
point_l2, _ = cv2.projectPoints(point2, Rl, tl, cameraMatrix=camera_matrix1_real, distCoeffs=dsitcoeffs1_real)
point_l2 = undistorted_point(point_l2, True)
# print(point2_l)

# 右目第1个点对应的像素坐标
point_r1, _ = cv2.projectPoints(point1, R_real, T_real, cameraMatrix=camera_matrix2_real, distCoeffs=dsitcoeffs2_real)
point_r1 = undistorted_point(point_r1, False)
# print(point1_r)

# 右目第2个点对应的像素坐标
point_r2, _ = cv2.projectPoints(point2, R_real, T_real, cameraMatrix=camera_matrix2_real, distCoeffs=dsitcoeffs2_real)
point_r2 = undistorted_point(point_r2, False)
# print(point2_r)


# 取第一张图的第一个点和最后一张图的第一个点
corners_L1 = read_points_from_file(point_path + "\\00_L.xml")
corners_L2 = read_points_from_file(point_path + "\\44_L.xml")
corners_R1 = read_points_from_file(point_path + "\\00_R.xml")
corners_R2 = read_points_from_file(point_path + "\\44_R.xml")
corner_L1 = corners_L1[0]
corner_L1 = undistorted_point(corner_L1, True)
corner_L2 = corners_L2[0]
corner_L2 = undistorted_point(corner_L2, True)
corner_R1 = corners_R1[0]
corner_R1 = undistorted_point(corner_R1, False)
corner_R2 = corners_R2[0]
corner_R2 = undistorted_point(corner_R2, False)

# 重建对应世界坐标系下点的坐标
# 计算标定板中的点对应的距离是用corner_xx
# 计算空间中任取两点的间距时用point_xx
dst1 = cv2.triangulatePoints(
    projMatr1=P1,
    projMatr2=P2,
    projPoints1=point_l1,
    projPoints2=point_r1
)
point_3D_1 = dst1 / dst1[3]
# print(point_3D_1[:3].T)

dst2 = cv2.triangulatePoints(
    projMatr1=P1,
    projMatr2=P2,
    projPoints1=point_l2,
    projPoints2=point_r2
)
point_3D_2 = dst2 / dst2[3]
# print(point_3D_2[:3].T)

print(distance(point_3D_1, point_3D_2))


""""      重建三维平面        """
# 建立一个空间平面
obj_points = []
for i in range(-3, 3):
    for j in range(-5, 5):
        obj_points.append([100 * i, 100 * j, 1500])
obj_points = np.array(obj_points, dtype=np.float64)
# print(obj_points)

# 对应左目像素坐标
obj_points_l, _ = cv2.projectPoints(obj_points, Rl, tl, cameraMatrix=camera_matrix1_real, distCoeffs=dsitcoeffs1_real)

# 图像点去畸变
obj_points_l = undistorted_point(obj_points_l, True)

# 对应右目的像素坐标
obj_points_r, _ = cv2.projectPoints(obj_points, R_real, T_real, cameraMatrix=camera_matrix2_real,
                                    distCoeffs=dsitcoeffs2_real)

# 图像点去畸变
obj_points_r = undistorted_point(obj_points_r, False)

# 三件测量三维点
dst = cv2.triangulatePoints(
    projMatr1=P1,
    projMatr2=P2,
    projPoints1=obj_points_l,
    projPoints2=obj_points_r
)
dst_2D = dst / dst[3]
# dst_2D = np.array(dst_2D, dtype=np.float64)
# print(dst_2D.T)

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter3D(obj_points[:, 0], obj_points[:, 1], obj_points[:, 2], c="r", marker="^")
ax.scatter3D(dst_2D.T[:, 0], dst_2D.T[:, 1], dst_2D.T[:, 2], c="g", marker="*")
ax.set_xlabel('X label')  # 画出坐标轴
ax.set_ylabel('Y label')
ax.set_zlabel('Z label')
plt.show()