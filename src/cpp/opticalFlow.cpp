#include <string>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>

#include "DenseORB.hpp"
#include "robustMatcher.hpp"



using namespace cv;
using namespace std;

void opticalFlow(const string &im1, const string &im2, const string &dst, const int &H_W);

int main(int argc, char** argv){

    string im1 = argv[1];
    string im2 = argv[2];
    string dst = argv[3];
    int H_W = atoi(argv[4]);

    opticalFlow(im1, im2, dst, H_W);
}


void opticalFlow(const string &im1, const string &im2, const string &dst, const int& H_W){
    //Ptr<Feature2D> denseORB = TFG::DenseORB::create(2000);
    Ptr<Feature2D> denseORB = ORB::create(1000);
    vector<Scalar> colors;
    RNG rng;

    for(int i = 0; i < 100; i++){
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }

    Mat frame1 = imread(im1);
    //cv::resize(frame1, frame1, cv::Size(H_W, H_W), 0, 0, cv::INTER_AREA);
    vector<KeyPoint> kp1, kp2;
    vector<DMatch> matches;


    TFG::robustMatcher rm(denseORB);
    vector<Point2f> p1, p2;
    Mat mask = Mat::zeros(frame1.size(), frame1.type());
    int i = 0;

    Mat frame2 = imread(im2);
    //cv::resize(frame2, frame2, cv::Size(H_W, H_W), 0, 0, cv::INTER_AREA);
    rm.simpleMatch(frame1, frame2, matches, kp1, kp2);

    for(auto it = matches.begin(); it != matches.end(); ++it){
        p1.push_back(kp1[it->queryIdx].pt);
        p2.push_back(kp2[it->trainIdx].pt);
    }

    
    for(int i = 0; i < p1.size(); i++){
        float dist = cv::norm(p1[i] - p2[i]);
        float maxDist = sqrt(frame1.rows*frame1.rows)/4;
        if(dist > 5 && dist < maxDist ){
            line(mask, p1[i], p2[i], Scalar(255,255,255), 2);
            circle(mask, p2[i], 3, Scalar(127,127,127), -1);
        }
    }

    Mat img;
    add(frame1, mask, img);
    imwrite(dst+"img.png", img);
    imwrite(dst+"mask.png", mask);


}
