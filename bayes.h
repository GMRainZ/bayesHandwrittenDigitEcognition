#pragma once
#include<opencv2/opencv.hpp>
#include<fstream>
#include<string>
#include<vector>

class Bayes
{
	static const int tempSize = 32;
	static const int rectSize = 20;

	struct CTemp{
		cv::Mat numsImg;
		std::vector<double>probabilityMatrix;
		int tag;
	}numsTemplate[10];


public:

	const std::string trainFilename = "D:\\作业\\智能计算\\trainData.txt";
	const std::string predictFilename = "D:\\作业\\智能计算\\predictData.txt";


public:

	Bayes() {
		for (int i = 0; i < 10; ++i)
		{
			numsTemplate[i].probabilityMatrix.assign(tempSize*tempSize,0);
			numsTemplate[i].tag = i;
			numsTemplate[i].numsImg = cv::Mat::ones(tempSize, tempSize, CV_8UC1);
		}
		
	}
	void greyToBinary(cv::Mat&);
	
	void train(cv::Mat&,const std::string&);
	
	void boundery(cv::Mat&);
	
	
	void binaryToTxt(cv::Mat&,const std::string&);


	void establishBayesianTemplate(const std::string&);

	void checkBayesianTemplate();
	void checkBayesianProbabilityMatrix();


	void preparePredictData(cv::Mat&,const std::string&);
	void predictHandewrittenDigit(const std::string&);
};

