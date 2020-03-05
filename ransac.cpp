#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace cv;
Mat srcImage;

//============================================== mark image =======================================================
Mat mark_img(Mat rgb, Mat hsv)
{
	Mat mark_image, hsvImage, dst;
	Mat whiteImage, yellowImage;

	Scalar lower_white(200,200,200);
	Scalar upper_white(255,255,255);
	inRange(rgb, lower_white, upper_white, whiteImage); //white 행렬이 lower<= a <= uppere 사이에 있으면 255 / 아니면 0 --> whiteImage에 저장 -> 흰색 부분빼곤 다 검정

	Scalar lower_shadow = Scalar(20,0,100);  
	Scalar upper_shadow = Scalar(100,100,255);
	inRange(hsv, lower_shadow, upper_shadow, hsvImage);
	
	bitwise_or(whiteImage, hsvImage, mark_image); //그림자 포함해서 mark_image만듬 (위에 흰점들은 뭘까)
	

	Scalar lower_yellow = Scalar(20,100,100);
	Scalar upper_yellow = Scalar(35,255,255);
	inRange(hsv, lower_yellow, upper_yellow, yellowImage); // 노랑부분을 추출하기 위한 임계값 추출

	bitwise_or(yellowImage, mark_image, mark_image); //그림자포함흰색이랑 노랑이랑 합연산 : 마스크
	Canny(mark_image,dst,50,150);
	srcImage.copyTo(dst,mark_image); //src에서 dst로 마스크 복사
	return dst;
}


//============================================== set roi =========================================================
Mat region_of_interest(Mat a, const Point **pts, Scalar color = Scalar(255,255,255))
{
	
	Mat temp(a.size(), a.type(),Scalar(0,0,0)); //a랑 똑같은 사이즈의 행렬 생
	Mat roi_image;
	
	int npt[] = {4};

	fillPoly(temp, pts, npt, 1, color); //비어있는 temp에 관심영역만큼 color채우기
	bitwise_and(temp, a, roi_image); //temp와 a를 and연산해서 roi_image에 저장 --> 원본에서 관심영역인곳 빼고 다 검은색됨 , 관심영역부분은 원본 동일
	return roi_image;
}


//============================================== ransac ==========================================================
void ransac(const vector<Point> &line, Mat& temp_Image)
{
	int N = 20, count = 0;
	double T = 3;
	srand((unsigned int)time(NULL));
	int length = line.size();
	Point num1 = line[length-1]; //y값 제일 큰거 : 화면 맨아래
	Point num2, num3, num4 = Point(0,0);

//=============================================2차-======================================================
	int c_max2 = 0;

	Mat best_model2(3,1, CV_64FC1); //abc 계수 모음
		
	Mat Org_X2(length, 3, CV_64FC1); //x^2, x, 1
	Mat Org_Y2(length, 1, CV_64FC1); //y

	for(int i=0; i<length; ++i) //근본 모든X,Y행렬 초기화
	{
		Org_X2.at<double>(i,0) = line[i].x * line[i].x;
		Org_X2.at<double>(i,1) = line[i].x;
		Org_X2.at<double>(i,2) = 1;
		Org_Y2.at<double>(i,0) = line[i].y;
	}
	
	
	for(int i=0; i<N; ++i)
	{
		num2 = line[rand() % (length-1)];
		//num2 = line[0];
		num4 = line[rand() % (length-1)];  //2.점추출

		vector<Point> ex2;
		ex2.push_back(num1); ex2.push_back(num2); ex2.push_back(num4);
	
		Mat X2(3,3, CV_64FC1); Mat Y2(3,1,CV_64FC1); Mat model2(3,1,CV_64FC1); Mat X_pinv2(3,3,CV_64FC1); //뽑은 점들 행렬 초기화 -> model찾을라고
	
		for(int i=0; i<3; ++i)
		{
			X2.at<double>(i,0) = (ex2[i].x)*(ex2[i].x);
			X2.at<double>(i,1) = ex2[i].x;
			X2.at<double>(i,2) = 1;
			Y2.at<double>(i,0) = ex2[i].y;
		}
	
		invert(X2, X_pinv2, DECOMP_SVD); 
		model2 = X_pinv2 * Y2;         //3. 방정식 구하기 AX=B 에서 A역행렬 곱함
		
		Mat residual2(length, 1, CV_64FC1);
		residual2 = abs(Org_Y2 - Org_X2*model2);   //4. residual 구하기 / 원래 y값에서 방정식y값 빼고 절대값
		
		for(int i=0; i<length; ++i)
			if(residual2.at<double>(i,0) <= T) ++count;
			
		if(count > c_max2) 
		{
			best_model2 = model2;
			c_max2 = count;
		}
		count = 0;
	}
	
	
	
	
//=========================================================== 3차 ==================================================
	
	int c_max3 = 0;

	Mat best_model3(4,1, CV_64FC1);

	
	Mat Org_X3(length, 4, CV_64FC1);
	Mat Org_Y3(length, 1, CV_64FC1);
	for(int i=0; i<length; ++i)
	{
		Org_X3.at<double>(i,0) = line[i].x * line[i].x*line[i].x;
		Org_X3.at<double>(i,1) = line[i].x*line[i].x;
		Org_X3.at<double>(i,2) = line[i].x;
		Org_X3.at<double>(i,3) = 1;
		Org_Y3.at<double>(i,0) = line[i].y;
	}
	
	
	
	for(int i=0; i<N; ++i)
	{
		num2 = line[rand() % (length-1)];
		//num2 = line[0];
		num3 = line[rand() % (length-1)];
		num4 = line[rand() % (length-1)];  //2.점추출
		vector<Point> ex3;
		ex3.push_back(num1); ex3.push_back(num2);ex3.push_back(num3); ex3.push_back(num4);
	
		Mat X3(4,4, CV_64FC1); Mat Y3(4,1,CV_64FC1); Mat model3(4,1,CV_64FC1); Mat X_pinv3(4,4,CV_64FC1);
	
		for(int i=0; i<4; ++i)
		{
			X3.at<double>(i,0) = (ex3[i].x)*(ex3[i].x)*(ex3[i].x);
			X3.at<double>(i,1) = (ex3[i].x)*(ex3[i].x);
			X3.at<double>(i,2) = (ex3[i].x);
			X3.at<double>(i,3) = 1;
			Y3.at<double>(i,0) = ex3[i].y;
		}
	
		invert(X3, X_pinv3, DECOMP_SVD); 
		model3 = X_pinv3 * Y3;         //3. 방정식 구하기
		
		Mat residual3(length, 1, CV_64FC1);
		residual3 = abs(Org_Y3 - Org_X3 * model3);   //4. residual 구하기
		
		for(int i=0; i<length; ++i)
			if(residual3.at<double>(i,0) <= T) ++count;
			
		if(count > c_max3) 
		{
			best_model3 = model3;
			c_max3 = count;
		}
		count = 0;
	}
	
	
	
	
//===============================================1차===============================================================
	
	
	int c_max = 0;

	Mat best_model(2,1, CV_64FC1); //a,b
	Mat Org_X(length, 2, CV_64FC1); //x,1
	Mat Org_Y(length, 1, CV_64FC1);

	for(int i=0; i<length; ++i) 
	{
		Org_X.at<double>(i,0) = line[i].x;
		Org_X.at<double>(i,1) = 1;
		Org_Y.at<double>(i,0) = line[i].y;
	}
	
	
	
	for(int i=0; i<N; ++i)
	{
		num2 = line[rand() % (length-1)]; //랜덤 점 추출
		//num2 = line[0];
		vector<Point> ex;
		ex.push_back(num1); ex.push_back(num2); // 점 두개 벡터에 넣음
	
		Mat X(2,2, CV_64FC1); Mat Y(2,1,CV_64FC1); Mat model(2,1,CV_64FC1); Mat X_pinv(2,2,CV_64FC1); // X: 점 대입값 (x,1) , Y:점 결과값(y), model:계수
	
		for(int i=0; i<2; ++i)
		{
			X.at<double>(i,0) = ex[i].x;
			X.at<double>(i,1) = 1;
			Y.at<double>(i,0) = ex[i].y;
		}
	
		invert(X, X_pinv, DECOMP_SVD); 
		model = X_pinv * Y;         //3. 방정식 구하기
		
		Mat residual(length, 1, CV_64FC1);
		residual = abs(Org_Y - Org_X*model);   //4. residual 구하기
		
		for(int i=0; i<length; ++i)
			if(residual.at<double>(i,0) <= T) ++count;
			
		if(count > c_max) 
		{
			best_model = model;
			c_max = count;
		}
		count = 0;
	}


//========================================가장 적절한거 뽑아서 그리기=======================================
	
	int best_c_max = c_max<c_max2 ? (c_max2<c_max3? c_max3:c_max2) : (c_max < c_max3? c_max3:c_max);
	if(best_c_max == c_max){ //1차 그리기
		cout<<"1차"<<endl;
		for(int i=0; i<length; ++i){
			int x = line[i].x;
			int y = best_model.at<double>(0,0)*x + best_model.at<double>(1,0);
			circle(temp_Image,Point(x,y),1,Scalar(255,0,0),2);	
		}
	}
	else if(best_c_max == c_max2){ //2차 그리기
		cout<<"2차"<<endl;
		for(int i=0; i<length; ++i){
		int x = line[i].x;
		int y = best_model2.at<double>(0,0)*x*x + best_model2.at<double>(1,0)*x + best_model2.at<double>(2,0);
		circle(temp_Image,Point(x,y),1,Scalar(255,0,0),2);	
		}
	}
	else{ //3차 그리기
		cout<<"3차"<<endl;
		for(int i=0; i<length; ++i){
		int x = line[i].x;
		int y = best_model3.at<double>(0,0)*x*x*x + best_model3.at<double>(1,0)*x*x + best_model3.at<double>(2,0)*x + best_model3.at<double>(3,0);
		circle(temp_Image,Point(x,y),1,Scalar(255,0,0),2);	
		}
	}
	
		
}


bool Compare(const Point i, const Point j)
{
	return i.y < j.y;

}

//============================================== main =============================================================
int main()
{	
	char buf[256];
	VideoCapture videoCapture("solidWhiteRight.mp4");

	if (!videoCapture.isOpened())
	{
		cout << "동영상 파일을 열수 없습니다. \n" << endl;

		char a;
		cin >> a;

		return 1;
	}

	videoCapture.read(srcImage);
	if (srcImage.empty()) return -1;

	VideoWriter writer;
	int codec = VideoWriter::fourcc('X', 'V', 'I', 'D');  // select desired codec (must be available at runtime)
	double fps = 10.0;                          // framerate of the created video stream
	string filename = "./live.avi";             // name of the output video file
	writer.open(filename, codec, fps, srcImage.size(), CV_8UC3);
	// check if we succeeded
	if (!writer.isOpened()) {
		cerr << "Could not open the output video file for write\n";
		return -1;
	}


	videoCapture.read(srcImage);
	int width = srcImage.size().width;
	int height = srcImage.size().height;

	

	while(1){
	videoCapture.read(srcImage);
		if (srcImage.empty()) break;



	//srcImage = imread("11.jpg");
	Mat dstImage, hsvImage, yellow_line, roi1, roi2;	
	//int height = srcImage.rows;
	//int width = srcImage.cols;	

	
	cvtColor(srcImage, hsvImage, COLOR_BGR2HSV); //srcImage를 hsv로 바꿔줌

	Point pts1[4] = {Point(10, height), Point(width*0.4, height*0.6), 
	                 Point(width*0.6, height*0.6), Point(width-10, height)}; //관심영역 Point 배열
	const Point *polygon[1] = {pts1};

	roi1 = region_of_interest(srcImage, polygon); //src이미지 관심영역
	roi2 = region_of_interest(hsvImage, polygon); //hsv이미지 관심영역
	
	dstImage = mark_img(roi1, roi2); //관심영역의 흰색노랑부분만 나옴 , 나머지는 검정
	GaussianBlur(dstImage, dstImage, Size(9,9), 1.0); //블러필터 : 잡음제거, 영상 부드럽게함
	
	Mat temp = region_of_interest(dstImage, polygon); //여기서 또 roi -> 위에 흰점들없어짐

	vector<Point> right_line, left_line;
	vector<Point>::iterator iter;
	
	for(int i=0; i<temp.rows; i++){ //왼쪽 차선좌표 벡터에 저장
		for(int j=0; j<temp.cols/2; j++){
			if(temp.at<Vec3b>(i,j) != Vec3b(0,0,0)) //검은색 부분이 아니면
				left_line.push_back(Point(j,i)); //그 좌표 벡터에 넣음
		}
	}
	
	for(int i=0; i<temp.rows; i++){ //오른쪽 차선좌표 벡터에 저장
		for(int j=temp.cols/2; j<temp.cols; j++){
			if(temp.at<Vec3b>(i,j) != Vec3b(0,0,0)){
				right_line.push_back(Point(j,i));
			}
		}
	}
	
	sort(left_line.begin(), left_line.end(), Compare); //벡터를 y좌표로 오름차순 정렬
	sort(right_line.begin(), right_line.end(), Compare);
	
	//==============================bird eye view=======================================================
	Point2f src_vertices[4];
    src_vertices[0] = Point(width*0.4, height*0.6);
	src_vertices[1] = Point(width*0.6, height*0.6);
    src_vertices[2] = Point(width, height);
    src_vertices[3] = Point(0, height);

    Point2f dst_vertices[4];
    dst_vertices[0] = Point(0, 0);
    dst_vertices[1] = Point(width, 0);
    dst_vertices[2] = Point(width, height);
    dst_vertices[3] = Point(0, height);

	Mat M = getPerspectiveTransform(src_vertices, dst_vertices);

	Mat dst(height, width, CV_8UC3);
    warpPerspective(dstImage, dst, M, dst.size(), INTER_LINEAR, BORDER_CONSTANT);
	
	Point lf[4] = { Point(80,height),Point(240,0),Point(380,0), Point(240,height)};
	const Point *polygon2[1] = {lf};

	Point rt[4] = { Point(760,height),Point(550,0),Point(750,0), Point(910,height)};
	const Point *polygon3[1] = {rt};
	Mat rrl = region_of_interest(dst,polygon2);
	Mat rrr = region_of_interest(dst,polygon3);
	bitwise_or(rrl,rrr,rrr);

	Mat M_pinv;
	invert(M, M_pinv, DECOMP_SVD);
	warpPerspective(rrr, dst, M_pinv, rrr.size(), INTER_LINEAR, BORDER_CONSTANT);

	vector<Point> right_line3, left_line3;
	vector<Point>::iterator iter3;
	
	for(int i=0; i<dst.rows; i++){ //왼쪽 차선좌표 벡터에 저장
		for(int j=0; j<dst.cols/2; j++){
			if(dst.at<Vec3b>(i,j) != Vec3b(0,0,0)) //검은색 부분이 아니면
				left_line3.push_back(Point(j,i)); //그 좌표 벡터에 넣음
		}
	}
	
	for(int i=0; i<dst.rows; i++){ //오른쪽 차선좌표 벡터에 저장
		for(int j=dst.cols/2; j<dst.cols; j++){
			if(dst.at<Vec3b>(i,j) != Vec3b(0,0,0)){
				right_line3.push_back(Point(j,i));
			}
		}
	}
	ransac(left_line3,srcImage);
	ransac(right_line3,srcImage);
	


//=====================================================================================
	//ransac(right_line, srcImage);
	//ransac(left_line, srcImage);

	imshow("result", srcImage);
	writer << srcImage;

	int ckey = waitKey(1);
	if(ckey == 27) break;
	}
	return 0;

}




