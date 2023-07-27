1. 运行python calib_visualize.py。
    - 显示相机和标定板在空间中的位置。
    - 在board2cam_RT文件夹中保存每一块标定板到左右相机的RT
    - 在board_points文件夹中保存每一块标定板上的点在左右相机中的每一个圆心坐标
2. 运行gen_board.py
    - 根据300x240.svg生成board.png作为渲染的模型
3. 编译C++的calib_board_rendering.exe并以Release模式执行
    - 在render_result文件夹中生成左右相机拍摄标定板的图像