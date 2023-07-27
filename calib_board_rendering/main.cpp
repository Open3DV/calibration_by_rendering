#include "opencv2/opencv.hpp"
#include <sstream>
#include <fstream>


using namespace cv;
using namespace std;

void read_param_xml(string xml_path, Mat& camera_intrinsic_l, Mat& camera_dist_l, Mat& camera_intrinsic_r, Mat& camera_dist_r, Mat& R, Mat& T)
{
    cv::FileStorage fs_in(xml_path, cv::FileStorage::READ);
    fs_in["camera_intrinsic_l"] >> camera_intrinsic_l;
    fs_in["camera_dist_l"] >> camera_dist_l;
    fs_in["camera_intrinsic_r"] >> camera_intrinsic_r;
    fs_in["camera_dist_r"] >> camera_dist_r;
    fs_in["R"] >> R;
    fs_in["T"] >> T;
}

void render(string output_png, Mat rvec, Mat tvec, Mat intrinsic, Mat distortion)
{
    Mat board = imread("board.png", 0);
    Mat sum_img = cv::Mat(1240, 1624, CV_32SC1, Scalar(0));
    Mat n_img = cv::Mat(1240, 1624, CV_16UC1, Scalar(0));
    Mat result_img = cv::Mat(1240, 1624, CV_8UC1, Scalar(0));
    for (int y = 0; y < board.rows; y++)
    {
        if (y % 2400 == 0)
        {
            cout << y / 240 << "%" << endl;
        }
        Mat point3d = Mat(board.cols, 3, CV_32FC1);
        for (int x = 0; x < board.cols; x++)
        {
            float xx = x / 100.0;
            float yy = y / 100.0;
            point3d.at<float>(x, 0) = xx;
            point3d.at<float>(x, 1) = yy;
            point3d.at<float>(x, 2) = 0;
        }


        
        Mat point2d;
        
        projectPoints(point3d,
            rvec,
            tvec,
            intrinsic,
            distortion,
            point2d
        );

        for (int x = 0; x < board.cols; x++)
        {
            int dst_x = (int)(point2d.at<Point2f>(x, 0).x);
            int dst_y = (int)(point2d.at<Point2f>(x, 0).y);

            if (dst_y >= 0 && dst_y < 1240 && dst_x>0 && dst_x < 1624)
            {
                sum_img.at<int>(dst_y, dst_x) += board.at<unsigned char>(y, x);
                n_img.at<unsigned short>(dst_y, dst_x) += 1;
            }

        }

    }


    for (int y = 0; y < result_img.rows; y++)
    {
        for (int x = 0; x < result_img.cols; x++)
        {
            if (n_img.at<unsigned short>(y, x) == 0)
            {
                result_img.at<unsigned char>(y, x) = 127;
            }
            else
            {
                result_img.at<unsigned char>(y, x) = sum_img.at<int>(y, x) / n_img.at<unsigned short>(y, x);
            }
        }
    }
    imwrite(output_png, result_img);
}

int main()
{
    Mat rvec = Mat(3, 1, CV_32FC1, Scalar(0.0));
    Mat tvec = Mat(3, 1, CV_32FC1, Scalar(0.0));

    Mat camera_intrinsic_l, camera_dist_l, camera_intrinsic_r, camera_dist_r, R, T;
    read_param_xml("calib_param.xml", camera_intrinsic_l, camera_dist_l, camera_intrinsic_r, camera_dist_r, R, T);

    for (int i = 0; i < 45; i++)
    {
        std::ostringstream ss;
        ss << "board2cam_RT/" << setw(2) << setfill('0') << i << "_L.xml";
        string input_path = ss.str();
        ss.str("");
        ss << "render_result/" << setw(2) << setfill('0') << i << "_L.png";
        string output_path = ss.str();
        cout << output_path << endl;

        cv::FileStorage fs_in(input_path, cv::FileStorage::READ);
        fs_in["cam_board2L_rvec"] >> rvec;
        fs_in["cam_board2L_tvec"] >> tvec;

        cout << rvec << endl;
        cout << tvec << endl;

        render(output_path, rvec, tvec, camera_intrinsic_r, camera_dist_r);

    }

    for (int i = 0; i < 45; i++)
    {
        std::ostringstream ss;
        ss << "board2cam_RT/" << setw(2) << setfill('0') << i << "_R.xml";
        string input_path = ss.str();
        ss.str("");
        ss << "render_result/" << setw(2) << setfill('0') << i << "_R.png";
        string output_path = ss.str();
        cout << output_path << endl;

        cv::FileStorage fs_in(input_path, cv::FileStorage::READ);
        fs_in["cam_board2R_rvec"] >> rvec;
        fs_in["cam_board2R_tvec"] >> tvec;

        cout << rvec << endl;
        cout << tvec << endl;

        render(output_path, rvec, tvec, camera_intrinsic_r, camera_dist_r);

    }

 
}