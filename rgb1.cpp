#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main(){
	Mat srcImage = imread("5.png"); //이미지 저장
	Mat dst(srcImage.size(),srcImage.type()); //똑같은 사이즈 빈 행렬 선언
	Point pts[4] = {Point(50,dst.rows),Point(dst.cols/2-40,dst.rows/2+45), Point(dst.cols/2+50,dst.rows/2+45),Point(dst.cols-50,dst.rows)}; //관심영역 사다리꼴
	const Point* polygons[1] = {pts};
	int npts[1] = {4}; //꼭지점 개수?
	fillPoly(dst,polygons,npts,1,Scalar(0,0,255)); //dst에 사다리꼴 채우기 (빨간색)
	Mat temp;
	bitwise_and(srcImage,dst,temp);
	double th1 = threshold(temp, temp, 200,255, THRESH_BINARY);//임계값 200 넘으면 255로 설정
	for(int x=0;x<srcImage.cols;x++){
		for(int y=0;y<srcImage.rows;y++){
			if(temp.at<Vec3b>(y,x)==(Vec3b){0,0,255})
				srcImage.at<Vec3b>(y,x)=(Vec3b){0,0,255};
		}
	}
	imshow("dstimage",srcImage);
	waitKey();
	return 0;
}

