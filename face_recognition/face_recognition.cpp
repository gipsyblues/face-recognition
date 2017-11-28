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

MatrixXd image2Matrix(vector<string> paths);
MatrixXd meanFace(MatrixXd m); 
MatrixXd subtractMeanFace(MatrixXd m, MatrixXd mean);
MatrixXd getEigenvectors(MatrixXd A); 
void displayEigenfaces(MatrixXd U);

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
    for (int i = 0; i < col; i++) {
	for (int j = 0; j < row; j++) {
	    res(j, 0) += m(j, i);
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

MatrixXd getEigenvectors(MatrixXd A) {
    MatrixXd AT = A.transpose();
    MatrixXd L = AT * A;

    // compute eigenvectors of L
    Eigen::EigenSolver<MatrixXd> es (L);
    MatrixXd V = es.pseudoEigenvectors();

    MatrixXd U = A * V;
    return U;
}

void displayEigenfaces(MatrixXd U) {
    namedWindow("Display window", WINDOW_AUTOSIZE);
    moveWindow("Display window", 20, 20);
    
    for (int i = 0; i < U.cols(); i++) {
	Mat image = Mat(trow, tcol, CV_8UC1, 0.0);
	for (int j = 0; j < U.rows(); j++) {
	    image.at<uchar>(j / tcol, j % tcol) = U(j, i);
	}
	imshow("Display window", image);
	waitKey(0);
    }

}

int main(int argc, char** argv) {
    cout << "hello world" << endl;

    MatrixXd m (2, 2);
    m(0,0) = 5;
    m(1,0) = 3;
    m(0,1) = 4;
    m(1,1) = 13;
    cout << m << endl;

    MatrixXd ev (2, 2);
    Eigen::EigenSolver<MatrixXd> es (m);

    for (int i = 0; i < ev.rows(); i++) {
	for (int j = 0; j < ev.cols(); j++) {
	    ev(i, j) = es.eigenvectors().col(j)[i].real();
	}
    }
    cout << ev << endl;

    cout << es.pseudoEigenvectors();


    vector<string> trainings  { "../training_images/subject01.normal.jpg",
				"../training_images/subject02.normal.jpg",
			        "../training_images/subject03.normal.jpg",
				"../training_images/subject07.normal.jpg",
				"../training_images/subject10.normal.jpg",
				"../training_images/subject11.normal.jpg",
				"../training_images/subject14.normal.jpg",
				"../training_images/subject15.normal.jpg"};
    
    // load images and create matrix
    MatrixXd matrix = image2Matrix(trainings);
    // compute the mean face
    MatrixXd meanface = meanFace(matrix); 
    // subtract mean face from each vector in the matrix
    MatrixXd A = subtractMeanFace(matrix, meanface);
    // compute eigenvectors from the matrix A
    MatrixXd U = getEigenvectors(A);
    cout << A << endl;
    displayEigenfaces(U);
}
