//
// Created by victor on 30/7/20.
//

#pragma once
#include <opencv2/features2d.hpp>
#include "ORBextractor.hpp"

/*
 * Wrapper para que el ORB extractor de ORBSlam sea clase hija de feature2d
 */
namespace TFG{

    class DenseORB : public cv::Feature2D {
    private:
        ORB_SLAM2::ORBextractor keypointExtractor = ORB_SLAM2::ORBextractor(1000, 1.2f, 8, 20, 7);

        cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;


    public:
        DenseORB( int nFeatures, float scaleFactor = 1.2f, int nLevels = 8, int iniThFAST = 20, int minThFAST = 7);

        static cv::Ptr<DenseORB> create(int nFeatures, float scaleFactor = 1.2f, int nLevels = 8, int iniThFAST = 20, int minThFAST = 7);

        void
        detect(const cv::_InputArray &image, std::vector<cv::KeyPoint> &keypoints, const cv::_InputArray &mask = cv::noArray()) override;

        void compute(const cv::_InputArray &image, std::vector<cv::KeyPoint> &keypoints,
                     const cv::_OutputArray &descriptors) override;

        void
        detectAndCompute(const cv::_InputArray &image, const cv::_InputArray &mask, std::vector<cv::KeyPoint> &keypoints,
                         const cv::_OutputArray &descriptors, bool useProvidedKeypoints) override;


    };

}


