/**
 * Computer Vision
 * Project 2: Face Recognition Using Eigenface Method
 * Shang-Hung Tsai
 */

#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui.hpp>

#include<iostream>
#include<string>
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>

using namespace cv;
using namespace std;
using Eigen::MatrixXd;

#define DISPLAY false
#define OUTPUT_IMAGE false

struct ClassificationResult {
    int classification;
    double distance;
};

MatrixXd image2Matrix(vector<string> paths);
MatrixXd meanFace(MatrixXd m); 
MatrixXd subtractMeanFace(MatrixXd m, MatrixXd mean);
MatrixXd getEigenvectors(MatrixXd A); 
void displayFace(MatrixXd U, string name);
vector<MatrixXd> getFeatureCoefficient(MatrixXd U, MatrixXd A); 
vector<MatrixXd> matrix2Vectors (MatrixXd matrix); 
vector<MatrixXd> reconstructFace (MatrixXd U, vector<MatrixXd> testsFC); 
vector<double> getDistance(vector<MatrixXd> a, vector<MatrixXd> b); 
vector<ClassificationResult> classify(vector<MatrixXd> FC, vector<MatrixXd> testFC); 
void showClassification(vector<ClassificationResult> CR, vector<double> d0, vector<string> trainings, vector<string> tests, double T0, double T1); 

int trow = 231;	    // height and width of training images
int tcol = 195;

/*
 * This function takes a vector of n image file paths,
 * loads all images (m pixels each), and store them in
 * a m x n matrix.
 */
MatrixXd image2Matrix(vector<string> paths) {
    int imgN = paths.size();
    string path = paths[0];
    Mat image = imread(path, IMREAD_GRAYSCALE);
    int length = image.rows * image.cols;

//    namedWindow("Display window", WINDOW_AUTOSIZE);
//    moveWindow("Display window", 20, 20);

    MatrixXd res (length, imgN);
    for (int i = 0; i < imgN; i++) {
	path = paths[i];
	image = imread(path, IMREAD_GRAYSCALE);
	for (int j = 0; j < length; j++) {
	    res(j, i) = image.at<uchar>(j / image.cols, j % image.cols);
	}
//	imshow("Display window", image);
//	waitKey(0);
    }
    return res;
}

/*
 * This function takes a matrix (m x n),
 * and returns a vector of the mean face (m x 1).
 */
MatrixXd meanFace(MatrixXd m) {
    int row = m.rows();
    int col = m.cols();
    MatrixXd res (row, 1);  
    // sum up the value in each row
    for (int i = 0; i < row; i++) {
	for (int j = 0; j < col; j++) {
	    res(i, 0) += m(i, j);
	}
    }
    // divide by col to get mean
    for (int i = 0; i < row; i++) {
	res(i, 0) = res(i, 0) / col;
    }
    return res;
}

/*
 * This function subtracts mean vector from each vector
 * in the matrix.
 */
MatrixXd subtractMeanFace(MatrixXd m, MatrixXd mean) {
    int row = m.rows();
    int col = m.cols();
    for (int i = 0; i < row; i++) {
	for (int j = 0; j < col; j++) {
	    m(i, j) = m(i, j) - mean(i, 0);
	}
    }
    return m;
}

/*
 * This function computes Eigenvectors of the input matrix.
 */
MatrixXd getEigenvectors(MatrixXd A) {
    MatrixXd AT = A.transpose();
    MatrixXd L = AT * A;

    // compute eigenvectors of L
    Eigen::EigenSolver<MatrixXd> es (L);
    MatrixXd V = es.pseudoEigenvectors();
    MatrixXd U = A * V;
    return U;
}

/*
 * This function takes eigenvectors matrix as input,
 * and display and save each row as a image.
 * The second argument is the name of output file(s).
 * Note that we normalize the vector values to 0-255.
 */
void displayFace(MatrixXd U, string name) {
    namedWindow("Display window", WINDOW_AUTOSIZE);
    moveWindow("Display window", 20, 20);
    
    for (int i = 0; i < U.cols(); i++) {
	double maxU = DBL_MIN;
	double minU = DBL_MAX;
	for (int j = 0; j < U.rows(); j++) {
	    maxU = max(maxU, U(j, i));
	    minU = min(minU, U(j, i));
	}

	Mat image = Mat(trow, tcol, CV_8UC1, 0.0);
	for (int j = 0; j < U.rows(); j++) {
	    image.at<uchar>(j / tcol, j % tcol) = (uchar) ((U(j, i) - minU) / (maxU - minU) * 255);
	}
	if (DISPLAY) {
	    imshow("Display window", image);
	    waitKey(0);
	}
	if (OUTPUT_IMAGE) {
	    imwrite("../output_images/" + name + to_string(i) + ".jpg", image);
	}
    }
}

/*
 * This function computes feature coefficient for training faces.
 * Matrix U is the eigenvectors, and A is the training vectors.
 */
vector<MatrixXd> getFeatureCoefficient(MatrixXd U, MatrixXd A) {
    int imgN = A.cols();
    vector<MatrixXd> fc;
    MatrixXd UT = U.transpose();
    for (int i = 0; i < imgN; i++) {
	MatrixXd Ri = A.col(i);
	fc.push_back(UT * Ri);
    }

    cout << "Here are the PCA cooefficient for each image:" << endl;
    for (int i = 0; i < imgN; i++) {	
	cout << "Image " + to_string(i) << endl;
	cout << fc[i] << endl << endl;
    }

    return fc;
}

/*
 * This function converts a matrix to a list of vectors (vertically).
 */
vector<MatrixXd> matrix2Vectors (MatrixXd matrix) {
    vector<MatrixXd> v;
    for (int i = 0; i < matrix.cols(); i++) {
	v.push_back(matrix.col(i));
    }
    return v;
}

/*
 * This function reconstruct input face images from eigenfaces.
 * U is the eigenfaces matrix, testsFC is a vector of all input faces.
 */
vector<MatrixXd> reconstructFace (MatrixXd U, vector<MatrixXd> testsFC) {
    vector<MatrixXd> RFaces;
    for (int i = 0; i < testsFC.size(); i++) {
	RFaces.push_back(U * testsFC[i]);
    }
    return RFaces;
}

/*
 * Given two lists of vectors a and b, this function computes 
 * pair-wise euclidean distance, and return them as an vector.
 * a and b are N * 1 vectors.
 */
vector<double> getDistance(vector<MatrixXd> a, vector<MatrixXd> b) {
    vector<double> res;
    for (int i = 0; i < a.size(); i++) {
	MatrixXd ai = a[i];
	MatrixXd bi = b[i];
	double diff = 0;
	for (int j = 0; j < ai.rows(); j++) {
	    diff += pow(ai(j, 0) - bi(j, 0), 2);
	}
	res.push_back(sqrt(diff));
    }    
    return res;
}

/*
 * Compute distance between input face and training images in the face space,
 * and classify each input face into one of the training images where the 
 * distance in feature coefficient is minimal.
 */
vector<ClassificationResult> classify(vector<MatrixXd> FC, vector<MatrixXd> testFC) {
    vector<ClassificationResult> res;
    for (int i = 0; i < testFC.size(); i++) {
	cout << "Compute di for test image " + to_string(i) << endl;
	int classification = -1;
	double minDiff = DBL_MAX;
	for (int j = 0; j < FC.size(); j++) {
	    double diff = 0;
	    for (int k = 0; k < FC[j].rows(); k++) {  // compute distance
		diff += pow(testFC[i](k, 0) - FC[j](k, 0), 2);
	    }
	    diff = sqrt(diff);
	    cout << "d" << j << "=" << diff << endl;
	    if (diff < minDiff) {
		minDiff = diff;
		classification = j;
	    }
	}
	cout << endl;
	ClassificationResult cr = {
	    classification,
	    minDiff,
	};
	res.push_back(cr);
    }
    return res;
}

void showClassification(vector<ClassificationResult> CR, vector<double> d0, vector<string> trainings, vector<string> tests, double T0, double T1) {
    for (int i = 0; i < CR.size(); i++) {
	cout << tests[i].substr(15, tests[i].size()) << ":" << endl;
	cout << "d0 = " << d0[i] << endl;
	if (d0[i] < T0) {
	    cout << "Classify--> NOT a face image" << endl;
	} else if (CR[i].distance < T1) {
	    string subject = trainings[CR[i].classification];
	    subject = subject.substr(subject.find("subject"), 9);
	    cout << "Classify-->" << subject << "  " << "distance = " << CR[i].distance << endl;
	} else {
	    cout << "Classify--> Unknown face" << endl;
	}
	cout << endl << "--------------------------------------------------------------" << endl;
    }
}
    

int main(int argc, char** argv) {

    vector<string> trainings  { "../training_images/subject01.normal.jpg",
				"../training_images/subject02.normal.jpg",
			        "../training_images/subject03.normal.jpg",
				"../training_images/subject07.normal.jpg",
				"../training_images/subject10.normal.jpg",
				"../training_images/subject11.normal.jpg",
				"../training_images/subject14.normal.jpg",
				"../training_images/subject15.normal.jpg"};
    
    vector<string> tests  { "../test_images/apple1_gray.jpg",
			    "../test_images/subject01.centerlight.jpg",
			    "../test_images/subject01.happy.jpg",
			    "../test_images/subject01.normal.jpg",
			    "../test_images/subject02.normal.jpg",
			    "../test_images/subject03.normal.jpg",
			    "../test_images/subject07.centerlight.jpg",
			    "../test_images/subject07.happy.jpg",
			    "../test_images/subject07.normal.jpg",
			    "../test_images/subject10.normal.jpg",
			    "../test_images/subject11.centerlight.jpg",
			    "../test_images/subject11.happy.jpg",
			    "../test_images/subject11.normal.jpg",
			    "../test_images/subject12.normal.jpg",
			    "../test_images/subject14.happy.jpg",
			    "../test_images/subject14.normal.jpg",
			    "../test_images/subject14.sad.jpg",
			    "../test_images/subject15.normal.jpg"};

    // load training images and create matrix
    MatrixXd matrix = image2Matrix(trainings);
    // compute and display the mean face
    MatrixXd meanface = meanFace(matrix); 
    displayFace(meanface, "meanface");
    // subtract mean face from each vector in the matrix
    MatrixXd A = subtractMeanFace(matrix, meanface);
    // compute eigenvectors from the matrix A
    MatrixXd U = getEigenvectors(A);
    // display Eigenface
    displayFace(U, "eigenface");
    vector<MatrixXd> FC = getFeatureCoefficient(U, A);

    // load test images and create matrix
    MatrixXd testImages = image2Matrix(tests);
    // subtract mean face from each test image vector
    testImages = subtractMeanFace(testImages, meanface);
    displayFace(testImages, "test-mean");
    // compute test images' projection onto face space
    vector<MatrixXd> testsFC = getFeatureCoefficient(U, testImages);
    // reconstruct input faces from eigenfaces
    vector<MatrixXd> RtestImages = reconstructFace(U, testsFC);
    for (int i = 0; i < RtestImages.size(); i++) {
	displayFace(RtestImages[i], "reconstructed" + to_string(i) + "-");
    }
    vector<MatrixXd> OtestImages = matrix2Vectors(testImages);
    // compute distance between input face image and its reconstruction
    vector<double> d0 = getDistance(RtestImages, OtestImages);
    vector<ClassificationResult> CR = classify(FC, testsFC);
    double T0 = 1000000000000;
    double T1 = 86500000;
    showClassification(CR, d0, trainings, tests, T0, T1);
}
