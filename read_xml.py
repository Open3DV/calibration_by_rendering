import cv2
import numpy as np
from spatial_transform import get_rvec_Yaskawa, invert_transform, multiply_transform, get_rmtx, get_rvec

def read_param(xml_path):
    fs = cv2.FileStorage(xml_path, cv2.FileStorage_READ)

    camera_intrinsic_l = fs.getNode('camera_intrinsic_l').mat()
    camera_dist_l = fs.getNode('camera_dist_l').mat()
    camera_intrinsic_r = fs.getNode('camera_intrinsic_r').mat()
    camera_dist_r = fs.getNode('camera_dist_r').mat()
    R = fs.getNode('R').mat()
    T = fs.getNode('T').mat()

    fs.release()
    R = np.matrix(R)
    T = np.matrix(T)
    
    #这是一个虚拟的中间摄像头，其R和T都是右摄像头的一半
    middle_R = get_rmtx(get_rvec(R)/2)
    middle_T = T/2
    
    return camera_intrinsic_l, camera_dist_l, camera_intrinsic_r, camera_dist_r, R, T, middle_R, middle_T
    
    
if __name__ == '__main__':
    camera_intrinsic_l, camera_dist_l, camera_intrinsic_r, camera_dist_r, R, T, _, _ = read_param("calib_param.xml")



