#include <iostream>
#include<opencv2/opencv.hpp>
#include<cmath>
// dont forget, use g++ filename.cpp `pkg-config --cflags --libs opencv4` for compilation
using namespace cv;
using namespace std;

double calculateEnergy(const Vec3b& pixel1, const Vec3b& pixel2) {// to calculate gradient i.e. Rx2 + Gx2+ Bx2
    int rx = pixel1[0] - pixel2[0];
    int gx = pixel1[1] - pixel2[1];
    int bx = pixel1[2] - pixel2[2];
    return rx * rx + gx * gx + bx * bx;
}


double** computeEnergyMatrix(const Mat& inputImage) {// to compute energy matrix
    int height = inputImage.rows;
    int width = inputImage.cols;

    // Allocate memory for the energy matrix
    double** energyMatrix = new double*[height];
    for (int y = 0; y < height; ++y) {
        energyMatrix[y] = new double[width];
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Calculate x and y neighbors

            int xLeft;
            int xRight;
            int yBelow;
            int yAbove;


            if(y==0){ // handling 1st row
                if(x==0){
                    xLeft = width-1;
                    xRight = min(width - 1, x + 1);
                    yAbove = height-1;
                    yBelow = min(height - 1, y + 1);

                }
                else if(x==width-1){
                    xLeft = max(0, x - 1);
                    xRight = 0;
                    yAbove = height-1;
                    yBelow = min(height - 1, y + 1);

                }
                else{
                    xLeft = max(0, x - 1);
                    xRight = min(width - 1, x + 1);
                    yAbove = height-1;
                    yBelow = min(height - 1, y + 1);

                }
            }
            else if(y==height-1){//handling last row
                if(x==0){
                    xLeft = width-1;
                    xRight = min(width - 1, x + 1);
                    yAbove = max(0, y - 1);
                    yBelow = 0;

                }
                else if(x==width-1){
                    xLeft = max(0, x - 1);
                    xRight = 0;
                    yAbove = max(0, y - 1);
                    yBelow = 0;

                }
                else{
                    xLeft = max(0, x - 1);
                    xRight = min(width - 1, x + 1);
                    yAbove = max(0, y - 1);
                    yBelow = 0;

                }

            }
            else{ // for all other rows
                if(x==0){
                    xLeft = width-1;
                    xRight = min(width - 1, x + 1);
                    yAbove = max(0, y - 1);
                    yBelow = min(height - 1, y + 1);

                }
                else if(x==width-1){
                    xLeft = max(0, x - 1);
                    xRight = 0;
                    yAbove = max(0, y - 1);
                    yBelow = min(height - 1, y + 1);

                }
                else{
                    xLeft = max(0, x - 1);
                    xRight = min(width - 1, x + 1);
                    yAbove = max(0, y - 1);
                    yBelow = min(height - 1, y + 1);

                }
            }

          

            // Calculate energy using neighbors
            double energyX = calculateEnergy(inputImage.at<Vec3b>(y, xLeft), inputImage.at<Vec3b>(y, xRight));
            double energyY = calculateEnergy(inputImage.at<Vec3b>(yAbove, x), inputImage.at<Vec3b>(yBelow, x));

            // Total energy is the sum of x and y energies
            energyMatrix[y][x] = sqrt(energyX + energyY);
        }
    }

    return energyMatrix;
}

int* findVerticalSeam(double** energyMatrix, int height, int width) {// finding vertical seam
    double** costMatrix = new double*[height];
    for (int y = 0; y < height; ++y) {
        costMatrix[y] = new double[width];
        for (int x = 0; x < width; ++x) {
            costMatrix[y][x] = energyMatrix[y][x];
        }
    }

    for (int y = 1; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double minPrevCost = costMatrix[y - 1][x];
            if (x == 0) {
                minPrevCost = min(minPrevCost, costMatrix[y - 1][x + 1]);
            }
            else if (x == width - 1) {
                minPrevCost = min(minPrevCost, costMatrix[y - 1][x - 1]);
            }
            else{
                minPrevCost = min(min(minPrevCost, costMatrix[y - 1][x + 1]),costMatrix[y-1][x-1]);
            }
            costMatrix[y][x] += minPrevCost;
        }
    }
    //Backtracking step to get vertical seam
    int* seam = new int[height];
    int minCostIndex = 0;
    for (int x = 1; x < width; ++x) {
        if (costMatrix[height - 1][x] < costMatrix[height - 1][minCostIndex]) {
            minCostIndex = x;
        }
    }
    seam[height - 1] = minCostIndex;

    for (int y = height - 2; y >= 0; --y) {
        int x = seam[y + 1];
        int minX = x;
        if (x > 0 && costMatrix[y][x - 1] < costMatrix[y][minX]) {
            minX = x - 1;
        }
        if (x < width - 1 && costMatrix[y][x + 1] < costMatrix[y][minX]) {
            minX = x + 1;
        }
        seam[y] = minX;
    }

    for (int y = 0; y < height; ++y) {
        delete[] costMatrix[y];
    }
    delete[] costMatrix;

    return seam;
}


int* findHorizontalSeam(double** energyMatrix, int height, int width) { //finding horizontal seam
    int* seam = new int[width];

    // Transpose energy matrix to treat rows as columns
    double** transposedEnergyMatrix = new double*[width];
    for (int x = 0; x < width; ++x) {
        transposedEnergyMatrix[x] = new double[height];
        for (int y = 0; y < height; ++y) {
            transposedEnergyMatrix[x][y] = energyMatrix[y][x];
        }
    }

    seam = findVerticalSeam(transposedEnergyMatrix, width, height);

    for (int x = 0; x < width; ++x) {
        delete[] transposedEnergyMatrix[x];
    }
    delete[] transposedEnergyMatrix;

    return seam;
}


void removeVerticalSeam(Mat& inputImage, int* seam) { //removing vertical seam
    int height = inputImage.rows;
    int width = inputImage.cols - 1;

    for (int y = 0; y < height; ++y) {
        for (int x = seam[y]; x < width; ++x) {
            inputImage.at<Vec3b>(y, x) = inputImage.at<Vec3b>(y, x + 1);
        }
    }
    inputImage = inputImage.colRange(0, width);// select columns from 0 to width-1
}


void removeHorizontalSeam(Mat& inputImage, int* seam) { //removing horizontal seam
    int height = inputImage.rows - 1;
    int width = inputImage.cols;

    Mat newImage(height, width, inputImage.type());

    for (int y = 0; y < height; ++y) {// here basically we are coping old img to new img excluding the seam
        int seamX = seam[y];
        for (int x = 0; x < width; ++x) {
            if (x < seamX) {
                newImage.at<Vec3b>(y, x) = inputImage.at<Vec3b>(y, x);
            } else {
                newImage.at<Vec3b>(y, x) = inputImage.at<Vec3b>(y + 1, x);
            }
        }
    }

    inputImage = newImage;
}




int main(){
    Mat inputImg=imread("/home/sumit/Desktop/sample_input/sample1.jpeg");// provide complete input image path here

    double** energyMatrix=computeEnergyMatrix(inputImg);
    int newWidth, newHeight;
    cout << "Enter the new width: ";
    cin >> newWidth;
    cout << "Enter the new height: ";
    cin >> newHeight;

    // // Print energy matrix for demonstration purposes
    // for (int y = 0; y < inputImg.rows; ++y) {
    //     for (int x = 0; x < inputImg.cols; ++x) {
    //         cout << energyMatrix[y][x] << "\t";
    //     }
    //     cout << "\n";
    // }

    // Iterate to reduce width
    while (inputImg.cols > newWidth) {
       
        int height = inputImg.rows;
        int width = inputImg.cols;
        int* seam = findVerticalSeam(energyMatrix, height, width);

     
        removeVerticalSeam(inputImg, seam);

       
        delete[] seam;

        // Update energy matrix for the modified image
        energyMatrix = computeEnergyMatrix(inputImg);
    }



    // Reduce height iteratively
    while (inputImg.rows > newHeight) {
        int height = inputImg.rows;
        int width = inputImg.cols;
        int* seam = findHorizontalSeam(energyMatrix, height, width);

        removeHorizontalSeam(inputImg, seam);

        
        delete[] seam;

        // Update energy matrix for the modified image
        energyMatrix = computeEnergyMatrix(inputImg);
    }


    
    imshow("Output Window",inputImg);// to display output image
    waitKey(0);//wait for a keystroke
    destroyAllWindows();// to close the display window
    
    imwrite("output.jpg", inputImg);  // Replace with the desired output file path to write the output image


    // Clean up dynamic memory for energy matrix{shayad program can be made faster using this}
    for (int y = 0; y < inputImg.rows; ++y) {
        delete[] energyMatrix[y];
    }
    delete[] energyMatrix;



    return 0;
}
