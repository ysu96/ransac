#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

//Hough Transform 파라미터
float rho = 2; // distance resolution in pixels of the Hough grid
float theta = 1 * CV_PI / 180; // angular resolution in radians of the Hough grid
float hough_threshold = 15;	 // minimum number of votes(intersections in Hough grid cell)
float minLineLength = 10; //minimum number of pixels making up a line
float maxLineGap = 20;	//maximum gap in pixels between connectable line segments


//차선 색깔 범위 
Scalar lower_white = Scalar(200, 200, 200); //흰색 차선 (RGB)
Scalar upper_white = Scalar(255, 255, 255);
Scalar lower_yellow = Scalar(10, 100, 100); //노란색 차선 (HSV)
Scalar upper_yellow = Scalar(40, 255, 255);


Mat roi(Mat img_edges, Point *points)
{
	Mat img_mask(img_edges.size(), img_edges.type());
	//사이즈 똑같은 빈행렬 생성

	//Scalar ignore_mask_color = Scalar(255, 255, 255);
	const Point* ppt[1] = { points }; //사다리꼴
	int npt[] = { 4 }; //꼭지점 수

	//검정빈창에 사다리꼴 흰색으로 채움
	fillPoly(img_mask, ppt, npt, 1, Scalar(255, 255, 255), LINE_8);

	//returning the image only where mask pixels are nonzero
	Mat img_masked(img_edges.size(), img_edges.type());
	bitwise_and(img_edges, img_mask, img_masked); //원래이미지랑 흰사다리꼴이미지랑 and연산해서 리턴


	return img_masked;
}

void filter_colors(Mat _img_bgr, Mat &img_filtered)
{
	UMat img_bgr;
	_img_bgr.copyTo(img_bgr);
	UMat img_hsv, img_combine;
	UMat white_mask, white_image;
	UMat yellow_mask, yellow_image;

	
	//Filter white pixels
	inRange(img_bgr, lower_white, upper_white, white_mask); //white_mask에 흰색,검정만
	bitwise_and(img_bgr, img_bgr, white_image, white_mask);
	//mask 범위내 두 행렬의 비트연산 white_image에 저장
	cvtColor(img_bgr, img_hsv, COLOR_BGR2HSV); //bgr에서 hsv로 변환

	//Filter yellow pixels( Hue 30 )
	


	inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask); //노랑이만
	bitwise_and(img_bgr, img_bgr, yellow_image, yellow_mask);


	//Combine the two above images
	addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, img_combine);


	img_combine.copyTo(img_filtered);
}




int main(){
	Mat srcImage = imread("5.png");
	Mat dst(srcImage.size(), srcImage.type());
	int width = srcImage.cols;
	int height = srcImage.rows;

	Point points[4];
		points[0] = Point(width * 0.1, height);
		points[1] = Point(width * 0.4, height *0.5);
		points[2] = Point(width * 0.6, height*0.5);
		points[3] = Point(width * 0.9, height);

	
	
	filter_colors(srcImage,dst);
	//dst = roi(dst,points);
	imshow("fds",dst);
	waitKey();
	return 0;
}
