//
// Created by victor on 30/7/20.
//

#include "DenseORB.hpp"

TFG::DenseORB::DenseORB(int nFeatures, float scaleFactor, int nLevels, int iniThFAST, int minThFAST) {
    keypointExtractor = ORB_SLAM2::ORBextractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
    //descriptorExtractor = cv::xfeatures2d::FREAK::create();
    descriptorExtractor = cv::ORB::create(nFeatures);
}

cv::Ptr<TFG::DenseORB> TFG::DenseORB::create(int nFeatures, float scaleFactor, int nLevels, int iniThFAST, int minThFAST) {
    return cv::makePtr<TFG::DenseORB>(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);
}

void
TFG::DenseORB::detect(const cv::_InputArray &image, std::vector<cv::KeyPoint> &keypoints, const cv::_InputArray &mask) {
    cv::Mat trash;
    keypointExtractor(image, mask, keypoints, trash);
}

void TFG::DenseORB::compute(const cv::_InputArray &image, std::vector<cv::KeyPoint> &keypoints,
                            const cv::_OutputArray &descriptors) {
    descriptorExtractor->compute(image, keypoints, descriptors);
}

void TFG::DenseORB::detectAndCompute(const cv::_InputArray &image, const cv::_InputArray &mask,
                                     std::vector<cv::KeyPoint> &keypoints, const cv::_OutputArray &descriptors,
                                     bool useProvidedKeypoints) {
    cv::Mat trash;
    keypointExtractor(image, mask, keypoints, trash);
    descriptorExtractor->compute(image, keypoints, descriptors);

}

