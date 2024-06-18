#include "bayes.h"

using namespace std;
using namespace cv;

void Bayes::greyToBinary(cv::Mat&img)
{
	const int row = img.rows, col = img.cols;
	int r, c;

	for (r = 0; r < row; ++r)
	{
		for (c = 0; c < col; ++c)
		{
			if (img.at<uchar>(r, c) >= 127)img.at<uchar>(r, c) = 255;
			else img.at<uchar>(r, c) = 0;
		}
	}
}

void Bayes::train(cv::Mat&img, const std::string&filename)
{
	const int m = img.rows/rectSize, n = img.cols/rectSize;
	int i, j;

	Mat dst;

	for (i = 0; i < m; ++i)
	{
		for (j = 0; j < n; ++j)
		{
			img(Range(i * rectSize, i * rectSize + rectSize),
				Range(j * rectSize, j * rectSize + rectSize)).copyTo(dst);

			boundery(dst);
			//resize(dst, dst, Size(tempSize, tempSize));
			//imshow("dst", dst);
			//waitKey(0);

			binaryToTxt(dst, filename);
		}
	}
}

void Bayes::boundery(cv::Mat&img)
{
	//resize(img, img, Size(tempSize, tempSize));
	//greyToBinary(img);

	const int m = img.rows, n = img.cols;

	int i, j, leftBoundery, rightBoundery, upperBoundery, lowerBoundery;
	bool find = false;

	//find upperBoundery
	for (i = 0; i < m && !find; ++i)//row
	{
		for (j = 0; j < n && !find; ++j)
		{
			if (!img.at<uchar>(i, j))continue;
			find = true;
			upperBoundery = i - 1 < 0 ? 0 : i - 1;
		}
	}

	//find lowerBoundery
	find = false;
	for (i = m - 1; i >= 0 && !find; --i)
	{
		for (j = 0; j < n && !find; ++j)
		{
			if (!img.at<uchar>(i, j))continue;
			find = true;
			lowerBoundery = i+1>=m?m-1:i+1;
		}
	}

	//find leftBoundery
	find = false;
	for (j = 0; j < n && !find; ++j)
	{
		for (i = 0; i < m && !find; ++i)
		{
			if (!img.at<uchar>(i, j))continue;
			find = true;
			leftBoundery = j - 1 < 0 ? 0 : j - 1;
		}
	}

	//find rightBoundery
	find = false;
	for (j = n - 1; j >= 0 && !find; --j)
	{
		for (i = 0; i < m && !find; ++i)
		{
			if (!img.at<uchar>(i, j))continue;
			find = true;
			rightBoundery = j+1>=n?n-1:j+1;
		}
	}

	img = img(Range(upperBoundery, lowerBoundery), Range(leftBoundery, rightBoundery)).clone();
	resize(img, img, Size(tempSize, tempSize));
	//greyToBinary(img);
	//imshow("img", img);
	//waitKey(0);
}

void Bayes::binaryToTxt(cv::Mat&img,const string&filename)
{
	const int row = img.rows, col = img.cols;
	int r, c;
	ofstream infs(filename,ios::app);

	

	for (r = 0; r < row; ++r)
	{
		for (c = 0; c < col; ++c)
		{
			if (img.at<uchar>(r, c))infs<<'1';
			else infs<<'0';
		}
	}

	infs << '\n';
}

void Bayes::establishBayesianTemplate(const string& filename)
{
	const int n = tempSize * tempSize, m = 500;
	int i, j;

	ifstream ifm(filename);
	string str;

	double t;

	vector<int>recordOne(n);
	for (int k = 0; k < 10; ++k)
	{
		recordOne.assign(n, 0);
		for (i = 0; i < m; ++i)
		{
			getline(ifm, str);
			for (j = 0; j < n; ++j)
			{
				if (str[j] == '1')++recordOne[j];
			}
		}

		//计算类条件概率
		for (j = 0; j < n; ++j)
		{
			t=recordOne[j] * 1.0f / m;
			numsTemplate[k].probabilityMatrix[j] = t /*< 0.15 ? 0 : t*/;
		}

		//int times = 0;
		//for (const auto& p : numsTemplate[k].probabilityMatrix)
		//{
		//	cout << p << ' ';
		//	if (!(++times % tempSize))cout << '\n';

		//}
		

		
		for (i = 0; i < tempSize; ++i)
		{
			for (j = 0; j < tempSize; ++j)
			{
				
				numsTemplate[k].numsImg.at<uchar>(i,j)=
					numsTemplate[k].probabilityMatrix[i * tempSize + j] * 255;
			}
		}
	
		//imshow("numsImg", numsTemplate[k].numsImg);
		//waitKey(0);
	}



}

void Bayes::checkBayesianTemplate()
{
	Mat show;
	for (int k = 0; k < 10; ++k)
	{
		resize(numsTemplate[k].numsImg, show, Size(400, 400));
		imshow("numsImg", show);
		waitKey(0);
	}
}

void Bayes::checkBayesianProbabilityMatrix()
{
	int i;
	for (int k = 0; k < 10; ++k)
	{
		printf("数字%4d\n", k);
		i = 0;
		for (const auto& p : numsTemplate[k].probabilityMatrix)
		{
			
			cout << p << ' ';
			if (!(++i % tempSize))cout << endl;
		}
	}
}

void Bayes::preparePredictData(Mat&img, const std::string&filename)
{
	const int singleSize = 50;
	const int m = img.rows / singleSize, n = img.cols / singleSize;
	int i, j;

	Mat dst;

	for (i = 0; i < m; ++i)
	{
		for (j = 0; j < n; ++j)
		{
			img(Range(i * singleSize, i * singleSize + singleSize),
				Range(j * singleSize, j * singleSize + singleSize)).copyTo(dst);

			boundery(dst);
			//resize(dst, dst, Size(tempSize, tempSize));
			//imshow("dst", dst);
			//waitKey(0);

			binaryToTxt(dst, filename);
		}
	}

}

void Bayes::predictHandewrittenDigit(const std::string&filename)
{
	const int n = tempSize * tempSize, m = 20, kval = 3;
	ifstream ifsm(filename);
	int i, j, k, l, x;
	string str;
	vector<double>predict;
	int accurate[10] = { 0, }, nonaccurate[10] = { 0, }, reject[10] = { 0 };

	for (k = 0; k < 10; ++k)
	{
		memset(accurate, 0, sizeof accurate);
		memset(nonaccurate, 0, sizeof nonaccurate);
		memset(reject, 0, sizeof reject);
		for (i = 0; i < m; ++i)
		{
			cout << "当前数字为: " << k << '\t';

			predict.assign(10, 1.0);
			getline(ifsm, str);
			for (l = 0; l < n; ++l)
			{
				for (j = 0; j < 10; ++j)
				{
					predict[j] *= (str[l] == '1') ?
						(numsTemplate[j].probabilityMatrix[l] + 1.0f) :
						(2.0f - numsTemplate[j].probabilityMatrix[l]);
				}

			}

			//sort(predict.begin(), predict.end());

			auto y = max_element(predict.begin(), predict.end());
			if (*y < 100)
			{
				++reject[k];
				continue;
			}
			x = y - predict.begin();
			(x == k) ? ++accurate[k] : ++nonaccurate[k];
			cout << "预测为: " << x << endl;
		}
		cout <<"准确率为："<<accurate[k]*1.0f/m << endl;
		cout << "错误率为：" << nonaccurate[k] * 1.0f / m << endl;
		cout << "拒绝识别率为：" << reject[k] * 1.0f / m << endl;
	}
}




