#include<iostream>
#include<vector>
#include<string>
#include"bayes.h"


using namespace std;
using namespace cv;


int main()
{
	//Mat trainData = imread("D:\\作业\\智能计算\\digits.png", IMREAD_GRAYSCALE);
	//Mat predictData= imread("D:\\作业\\智能计算\\predict2.webp", IMREAD_GRAYSCALE);
	//if (trainData.empty() || predictData.empty())
	//{
	//	cerr << "there is an error in opening files" << endl;
	//	return -1;
	//}

	Bayes bayes;
	//bayes.greyToBinary(trainData);
	//bayes.greyToBinary(predictData);

	//bayes.train(trainData,bayes.trainFilename);

	bayes.establishBayesianTemplate(bayes.trainFilename);

	bayes.checkBayesianTemplate();
	//bayes.checkBayesianProbabilityMatrix();

	//bayes.preparePredictData(predictData, bayes.predictFilename);

	bayes.predictHandewrittenDigit(bayes.predictFilename);

	return 0;
}