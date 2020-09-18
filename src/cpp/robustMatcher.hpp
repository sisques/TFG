#pragma once

#include <opencv2/opencv.hpp>


namespace TFG{

    class robustMatcher {
    private:
        /*
         * Puntero al detector de features
         */
        cv::Ptr<cv::FeatureDetector> detector;
        /*
         * Puntero al extractor de descriptores
         */
        cv::Ptr<cv::DescriptorExtractor> descriptor;
        /*
         * Método de calculo de la distancia entre puntos de interés
         */
        int normType;
        /*
         * Ratio segundo vecino
         */
        float ratio;
        /*
         * Si es cierto, refina la fundamental
         */
        bool refineF;
        /*
         * Si es cierto, refina los matches
         */
        bool refineM;
        /*
         * Distancia mínima a la epipolar
         */
        double distance;
        /*
         * Nivel de confianza
         */
        double confidence;

        /*
         * Vectores que almacenan los inliers y outliers de las 2 imágenes a emparejar
         */
        std::vector<cv::KeyPoint> inliers1, outliers1, inliers2, outliers2;

        /*
         * Método encargado de calcular la matriz fundamental, y de refinar los matches calculados
         */
        cv::Mat ransacTest( const std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kp1,
                            std::vector<cv::KeyPoint>& kp2, std::vector<cv::DMatch>& outMatches,
                            std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2 );

    public:

        /*
         * Constructor de la clase, se puede especificar un extractor de descriptores distinto al extractor de puntos
         * de interés
         */
        robustMatcher(  const cv::Ptr<cv::FeatureDetector> &detector_,
                        const cv::Ptr<cv::DescriptorExtractor> &descriptor_ = cv::Ptr<cv::DescriptorExtractor>() );

        /*
         * Hace matching de los puntos de interes mediante RANSAC, devuelve matriz fundamental y los matches
         */
        cv::Mat match(  const cv::Mat& im1, const cv::Mat& im2, std::vector<cv::DMatch>& matches,
                        std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
                        std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2);

        /*
         * Hace matching de los puntos de interes mediante RANSAC, devuelve matriz fundamental y los matches, antes de
         * hacer el matching filtra los puntos de interés que caen dentro de las máscaras.
         */
        cv::Mat match(  const cv::Mat& im1, const cv::Mat& im2, const cv::Mat& mask1, const cv::Mat& mask2,
                        std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kp1,
                        std::vector<cv::KeyPoint>& kp2, std::vector<cv::Point2f>& p1,
                        std::vector<cv::Point2f>& p2);

        void simpleMatch(       const cv::Mat& im1,
                                const cv::Mat& im2,
                                std::vector<cv::DMatch>& matches,
                                std::vector<cv::KeyPoint>& kp1,
                                std::vector<cv::KeyPoint>& kp2      );


        std::vector<cv::KeyPoint> getInliers(int frame){
            if (frame==1){
                return inliers1;
            } else {
                return inliers2;
            }
        }

        std::vector<cv::KeyPoint> getOutliers(int frame){
            if (frame==1){
                return outliers1;
            } else {
                return outliers2;
            }
        }

    };


}

