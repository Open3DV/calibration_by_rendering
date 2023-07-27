import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import cv2 
from draw_tools import draw_axis, set_axes_equal, draw_camera, draw_camera_from_inverted_transform, draw_board, project_board
from spatial_transform import get_rvec_Yaskawa, invert_transform, multiply_transform, get_rmtx, get_rvec
import json
from read_xml import read_param

PI = 3.1415926
    

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # 读取相机内外参数
    camera_intrinsic_l, camera_dist_l, camera_intrinsic_r, camera_dist_r, R, T, middle_R, middle_T = read_param("calib_param.xml")
    
    # 左右相机到悬挂点的RT
    cam_R2L_rvec, cam_R2L_tvec = invert_transform(get_rvec(R), T)
    cam_M2L_rvec, cam_M2L_tvec = invert_transform(get_rvec(middle_R), middle_T)
    cam_L2M_rvec, cam_L2M_tvec = get_rvec(middle_R), middle_T
    cam_R2M_rvec, cam_R2M_tvec = multiply_transform(cam_L2M_rvec, cam_L2M_tvec, cam_R2L_rvec, cam_R2L_tvec)
    
    # 世界坐标系，相机正下方地面中心
    base_rvec = np.matrix([0,0,0], dtype=np.float32).T  
    base_tvec = np.matrix([0,0,0], dtype=np.float32).T
    draw_axis(ax, rvec=base_rvec, tvec=base_tvec)

    # 相机悬挂点RT
    mount2base_rvec = np.matrix([PI,0,0], dtype=np.float32).T  
    mount2base_tvec = np.matrix([0,0,1500], dtype=np.float32).T
    
    # 相机到地面中心的RT
    cam_L2base_rvec, cam_L2base_tvec = multiply_transform(mount2base_rvec, mount2base_tvec, cam_L2M_rvec, cam_L2M_tvec)
    cam_R2base_rvec, cam_R2base_tvec = multiply_transform(mount2base_rvec, mount2base_tvec, cam_R2M_rvec, cam_R2M_tvec)

    # 画出相机
    draw_camera(ax, rvec=cam_L2base_rvec, tvec=cam_L2base_tvec, color='r')
    draw_camera(ax, rvec=cam_R2base_rvec, tvec=cam_R2base_tvec, color='g')
    draw_axis(ax, rvec=cam_L2base_rvec, tvec=cam_L2base_tvec)
    draw_axis(ax, rvec=cam_R2base_rvec, tvec=cam_R2base_tvec)
    
    idx = 0
    for tx in range(-1, 2):
        for ty in range(-1, 2):
            board_trans_tvec = base_tvec + np.matrix([tx*500, ty*400, 0]).reshape([3,1])
            
            # 5个旋转角度
            board_pos_rvecs = [[0,0,0], [0.5,0,0], [-0.5,0,0], [0,0.5,0], [0,-0.5,0]]
            for i in range(5):
                board_pos_rvec = board_pos_rvecs[i]
                board_pos_rvec = np.matrix(board_pos_rvec, dtype=np.float32).reshape([3,1])
                
                # 将标定板中心移到base原点
                board_pos_tvec = np.matrix([-150, -120, 0], dtype=np.float32).reshape([3,1])
                
                # 标定板先平移到base原点，再旋转角度
                board_pos_rvec, board_pos_tvec = multiply_transform(board_pos_rvec, 
                                                                    np.matrix([0,0,0], dtype=np.float32).T, 
                                                                    np.matrix([0,0,0], dtype=np.float32).T, 
                                                                    board_pos_tvec)
                
                # 再在地面上水平平移到达标定位置
                board2base_rvec, board2base_tvec = multiply_transform(np.matrix([0,0,0], dtype=np.float32).T, 
                                                                      board_trans_tvec, 
                                                                      board_pos_rvec, 
                                                                      board_pos_tvec)
                
                draw_board(ax, rvec=board2base_rvec, tvec=board2base_tvec, color='r')
                
                # 算出相机到标定板的RT
                base2board_rvec, base2board_tvec = invert_transform(board2base_rvec, board2base_tvec)
                cam_L2board_rvec, cam_L2board_tvec = multiply_transform(base2board_rvec, base2board_tvec, cam_L2base_rvec, cam_L2base_tvec)
                cam_R2board_rvec, cam_R2board_tvec = multiply_transform(base2board_rvec, base2board_tvec, cam_R2base_rvec, cam_R2base_tvec)
                
                base2L_rvec, base2L_tvec = invert_transform(cam_L2base_rvec, cam_L2base_tvec)
                base2R_rvec, base2R_tvec = invert_transform(cam_R2base_rvec, cam_R2base_tvec)
                cam_board2L_rvec, cam_board2L_tvec = multiply_transform(base2L_rvec, base2L_tvec, board2base_rvec, board2base_tvec)
                cam_board2R_rvec, cam_board2R_tvec = multiply_transform(base2R_rvec, base2R_tvec, board2base_rvec, board2base_tvec)
                
                # 保存RT
                with open('board2cam_RT\%02d_L.txt'%idx, 'w') as fp:
                    fp.write(str(cam_board2L_rvec[0,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2L_rvec[1,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2L_rvec[2,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2L_tvec[0,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2L_tvec[1,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2L_tvec[2,0]))
                    fp.write('\n')
                with open('board2cam_RT\%02d_R.txt'%idx, 'w') as fp:
                    fp.write(str(cam_board2R_rvec[0,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2R_rvec[1,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2R_rvec[2,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2R_tvec[0,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2R_tvec[1,0]))
                    fp.write('\n')
                    fp.write(str(cam_board2R_tvec[2,0]))
                    fp.write('\n')
                    
                # 保存圆点坐标
                project_board('board_points\%02d_L.xml'%idx, cam_board2L_rvec, cam_board2L_tvec, camera_intrinsic_l, camera_dist_l)
                project_board('board_points\%02d_R.xml'%idx, cam_board2R_rvec, cam_board2R_tvec, camera_intrinsic_r, camera_dist_r)

                    
                idx+=1

    
    set_axes_equal(ax)
    plt.show()