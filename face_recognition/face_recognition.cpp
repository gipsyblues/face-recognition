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

using namespace cv;
using namespace std;
using Eigen::MatrixXd;

MatrixXd image2Matrix(vector<string> paths);
MatrixXd meanFace(MatrixXd m); 
MatrixXd subtractMeanFace(MatrixXd m, MatrixXd mean);

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

    MatrixXd res (length, imgN);
    for (int i = 0; i < imgN; i++) {
	path = paths[i];
	image = imread(path, IMREAD_GRAYSCALE);
	for (int j = 0; j < length; j++) {
	    res(j, i) = image.at<uchar>(j / image.cols, j % image.cols);
	}
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


int main(int argc, char** argv) {
    cout << "hello world" << endl;

    MatrixXd m (3, 2);
    m(0,0) = 3;
    m(1,0) = 2;
    m(0,1) = 1;
    m(1,1) = 0;
    cout << m << endl;
}
