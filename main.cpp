#include <iostream>
#include <random>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>



using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

unsigned const int CC = 2; // кол-во камер
string RESULT_DIR = "/home/gigachad/Документы/Code/Test/Result/";
string RESOURCE_DIR = "/home/gigachad/Документы/Code/Test/Resource/";
string IMGSL_DIR = "IMGSL/";
string IMGSR_DIR = "IMGSR/";
string CALIBS_DIR = "CALIBS_DIR/";
string FORMAT_IMG = ".png";
string FORMAT_TEXT = ".txt" ;
unsigned const int COUNT_IMGS = 1; // кол-во изображений


//генератор случайных чисел
int getRN(int min = 0, int max = 250){
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> rgb(1,255);
    return rgb(rng);
}


// нарисовать линию, заданую уравнение ax + by + c = 0 (a,b,c) на изображениии img
void drawEpiLine(Mat img, Point3f l){
    int c = img.cols;
    for(int i = 0; i < 3; i++){
        int x0 = 0;
        int y0 = -l.z/l.y;
        int x1 = c;
        int y1 = -(l.z + l.x*c)/l.y;

        Point p1(x0,y0), p2(x1,y1);
        
        line(img, p1, p2, cv::Scalar(getRN(),getRN(),getRN()),1,LINE_8);
    }

}

//чтение p1 p2 с файла
void readProgectionMatrix(const string filepath1, Mat &out_P1){
    double temp;
    int cols = 4;
    int rows = 3;
    std::ifstream f1(filepath1);
 
    if (!f1)
    {
        std::cout << "Файл не найден" << std::endl;
    }
    else{
        out_P1 = Mat::ones(rows,cols,CV_64F);
        for (int r = 0; r < rows; r++){
            for(int c = 0; c < cols; c++){
                f1>>temp;
                out_P1.at<double>(r,c) = temp;
            }

        }
    }

    f1.close();
}

// чтение колибровачных матриц
void readRKt(const string Kpath, const string Rpath, const string tpath,const string dpath,
                         Mat &K,Mat &R,Mat &t, Mat &d){
double temp;
    std::ifstream fK(Kpath);
    std::ifstream fR(Rpath);
    std::ifstream ft(tpath);
    std::ifstream fd(dpath);

 
    if (!fK || !fR || !ft || !fd)
    {
        std::cout << "Файлы не найдены" << std::endl;
    }
    else{
        K = Mat::ones(3,3,CV_64F);
        R = Mat::ones(3,3,CV_64F);
        t = Mat::ones(3,1, CV_64F);
        d = Mat::ones(5,1, CV_64F);

        for (int r = 0; r < 5; r++){
            fd>>temp;
            d.at<double>(r,0) = temp;
        }
        for (int r = 0; r < 3; r++){
            ft>>temp;
            t.at<double>(r,0) = temp;
            for(int c = 0; c < 3; c++){
                fK>>temp;
                K.at<double>(r,c) = temp;
                fR>>temp;
                R.at<double>(r,c) = temp;
            }

        }
        cout<<"Данные успешно считаны"<<endl;
    }

    fK.close();
    fR.close();
    ft.close();
    fd.close();

}

//нахождение фундаментальной матрицы из 2 progective matrix
void fundamentalFromProjections( const Mat P1,
                              const Mat P2,
                              Mat &F )
{

    Mat_<double> X[3];
    vconcat( P1.row(1), P1.row(2), X[0] );
    vconcat( P1.row(2), P1.row(0), X[1] );
    vconcat( P1.row(0), P1.row(1), X[2] );

    Mat_<double> Y[3];
    vconcat( P2.row(1), P2.row(2), Y[0] );
    vconcat( P2.row(2), P2.row(0), Y[1] );
    vconcat( P2.row(0), P2.row(1), Y[2] );
    
    Mat_<double> XY;
    F = Mat(3,3, CV_64F);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
      {
        vconcat(X[j], Y[i], XY);
        F.at<double>(i,j) = determinant(XY);
      }
}

// нахождение угла между эпиполярной линией и горизонтальной линией
double getAngleLineHorizontal(double a,double b){
    double angle = acos( (double)b/(sqrt(pow(a,2)+pow(b,2))) ) * (double)180/CV_PI;
    if(b < 0)
        return 180 - angle;
    else   
        return angle;
}

// получени P из колибровочных матриц
void projectionFromKRt(const Mat K, const Mat R, const Mat t, Mat &P)
{
  hconcat( K*R, K*t, P );
}

// получение ключевых точек
void getKeyPoints(Mat imgL, Mat imgR, std::vector<Point2f> &kps1, std::vector<Point2f> &kps2,
                    string filename = "imgkp"){
    // поиск ключевых точек и дескрипторов к ним
	int minHessian = 400;
    Ptr<SURF> detector = SURF::create( minHessian );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( imgL, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( imgR, noArray(), keypoints2, descriptors2 );

    // поиск совподающих точек
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors1, descriptors2, knn_matches, 2 );

	const float ratio_thresh = 0.3f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    //формируем два набора соответвующих ключевых точек
    for(int i = 0; i < good_matches.size(); i++){
        Point2f kp1 = keypoints1[good_matches[i].queryIdx].pt;
        kps1.push_back(kp1);
        circle(imgL, kp1, 3, cv::Scalar(getRN(),getRN(),getRN()), 4);

        Point2f kp2 = keypoints2[good_matches[i].trainIdx].pt;
        kps2.push_back(kp2);
        circle(imgR, kp2, 3, cv::Scalar(getRN(),getRN(),getRN()), 4);
    }

    // нарисовать соответсвующие ключевые точки и соеденить линиями
    Mat img_matches;
    drawMatches( imgL, keypoints1, imgR, keypoints2, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
    //imshow("IMG MATCHES", img_matches);
    imwrite(RESULT_DIR+filename+FORMAT_IMG, img_matches);
}

// финальная функция выдающая средний угол уколона
double resultFunction(Mat imgL, Mat imgR, Mat K[CC], Mat R[CC], Mat t[CC], Mat P[CC] = nullptr,
                      string dir = "", string flL = "imgL", string flR = "imgR",
                      string flres = "res"){

    std::vector<Point2f> kps1, kps2;
    getKeyPoints(imgL, imgR, kps1, kps2);

    //считаем фундаментельную матрицу через ключевые точки
    Mat F;
    F = findFundamentalMat(kps1, kps2, noArray(), FM_LMEDS);

    //получаем P1 и P2 из калибровочных матриц
    // projectionFromKRt(K1,R1,t1,P1);
    // projectionFromKRt(K2,R2,t2,P2);
    //считаем фундоментальную матрицу через калибровочные матрицы
    // Mat F;
    // fundamentalFromProjections(P1, P2, F);

    // находим эпиполярные линиии для каждого изображения
    std::vector<Point3f> linesL, linesR;
    computeCorrespondEpilines(kps1, 1, F, linesR);
    computeCorrespondEpilines(kps2, 2, F, linesL);

    // найдем среднее отклонение линний от горизонтали 
    double sumAngles = 0;
    for(int i = 0; i < linesR.size(); i++){
        sumAngles += getAngleLineHorizontal(linesL[i].x, linesL[i].y);
    }

    //рисуем все эпиполярные линиии на каждом из изображений
    for(int i = 0; i < linesL.size(); i++){
        drawEpiLine(imgR, linesR[i]); 
        drawEpiLine(imgL, linesL[i]); 
    }

    // imshow("IMG L", imgL);
    // imshow("IMG R", imgR);
    imwrite(dir+flL+FORMAT_IMG, imgL);
    imwrite(dir+flR+FORMAT_IMG, imgR);

    double res = (double)sumAngles/linesR.size();

    //записываем результат в файл
    std::ofstream out;          
    out.open(RESULT_DIR+flres+FORMAT_TEXT); 
    if (out.is_open())
    {
        out << res << std::endl;
    }
    out.close();

    return res;
}

int main()
{
    Mat imgR, imgL;
    Mat P[CC], K[CC],R[CC],t[CC],d[CC];

    //проходим по всем изображениям в папке
    for(int i = 0; i < COUNT_IMGS; i++){

        //загрузка изображений
        std::string img1_path = RESOURCE_DIR+IMGSL_DIR+"imgL"+to_string(i)+FORMAT_IMG;
	    std::string img2_path = RESOURCE_DIR+IMGSR_DIR+"imgR"+to_string(i)+FORMAT_IMG;

        imgR = imread(img1_path, IMREAD_COLOR);
	    imgL = imread(img2_path, IMREAD_COLOR);

        if (imgR.empty() || imgL.empty() )
        {
            cout << "Изображение не найдено\n" << endl;
            return -1;
        }

        // загрузка внутренних и внешних параметров 
        for(int j = 0; j < CC; j++){
            readProgectionMatrix(RESOURCE_DIR+CALIBS_DIR+"P"+to_string(j+1)+"_"+to_string(i)+FORMAT_TEXT,
                                P[j]);
            readRKt(RESOURCE_DIR+CALIBS_DIR+"K"+to_string(j+1)+"_"+to_string(i)+FORMAT_TEXT,
                RESOURCE_DIR+CALIBS_DIR+"R"+to_string(j+1)+"_"+to_string(i)+FORMAT_TEXT,
                RESOURCE_DIR+CALIBS_DIR+"t"+to_string(j+1)+"_"+to_string(i)+FORMAT_TEXT,
                RESOURCE_DIR+CALIBS_DIR+"d"+to_string(j+1)+"_"+to_string(i)+FORMAT_TEXT,
                K[j],R[j],t[j],d[j]);

        }
        resultFunction(imgL, imgR, K,R,t,P,RESULT_DIR, "imgL"+to_string(i), "imgR"+to_string(i),
                        "res"+to_string(i));
    }

    waitKey();
    return 0;

}
