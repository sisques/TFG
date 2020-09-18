
#include "robustMatcher.hpp"
// Métodos privados
namespace TFG{

    cv::Mat robustMatcher::ransacTest( const std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kp1,
                        std::vector<cv::KeyPoint>& kp2, std::vector<cv::DMatch>& outMatches,
                       std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2 ){
        outMatches.clear();
        outliers1.clear();
        outliers2.clear();
        inliers1.clear();
        inliers2.clear();
        p1.clear();
        p2.clear();


        // 1. Convertir KeyPoints en Point2f
        std::vector<cv::KeyPoint> p1_, p2_;
        for(auto it = matches.begin(); it != matches.end(); ++it){
            // KP de la izda
            p1.push_back(kp1[it->queryIdx].pt);
            p1_.push_back(kp1[it->queryIdx]);
            // KP de la dcha
            p2.push_back(kp2[it->trainIdx].pt);
            p2_.push_back(kp2[it->trainIdx]);
        }

        // 2. Calcular la matriz fundamental usando RANSAC
        std::vector<uchar> mask(p1.size(), 0);
        cv::Mat fundamental = cv::findFundamentalMat(p1, p2, mask, cv::FM_RANSAC, distance, confidence);


        // 3. Obtener los inliers (matches restantes)
        auto it_mask = mask.begin();
        auto it_matches = matches.begin();
        for( ; it_mask != mask.end(); ++it_mask, ++it_matches){
            // Es inlier
            if(*it_mask){
                outMatches.push_back(*it_matches);
            }
        }

        for(int i = 0; i < mask.size(); i++){
            if(!mask[i]) {
                outliers1.push_back(p1_[i]);
                outliers2.push_back(p2_[i]);
            } else {
                inliers1.push_back(p1_[i]);
                inliers2.push_back(p2_[i]);
            }
        }

        // 4. Usa los matches calculados para re-estimar la matriz fundamental usando el algoritmo de 8 puntos
        if(refineF){
            p1.clear();
            p2.clear();


            // 4.1 Convertir KeyPoints en Point2f
            for(auto it = outMatches.begin(); it != outMatches.end(); ++it){
                // KP de la izda
                p1.push_back(kp1[it->queryIdx].pt);
                // KP de la dcha
                p2.push_back(kp2[it->trainIdx].pt);
            }
            // 4.2 Calcular la matriz fundamental mediante el algoritmo de 8 puntos


            auto _8pts= cv::findFundamentalMat(p1, p2, cv::FM_8POINT);
            if(_8pts.rows * _8pts.cols == 9){
                fundamental = _8pts;
            }
        }

        if(refineM){
            p1.clear();
            p2.clear();
            // 4.1 Convertir KeyPoints en Point2f
            for(auto it = outMatches.begin(); it != outMatches.end(); ++it){
                // KP de la izda
                p1.push_back(kp1[it->queryIdx].pt);
                // KP de la dcha
                p2.push_back(kp2[it->trainIdx].pt);
            }
            std::vector<cv::Point2f> np1, np2;

            // Corregir las coordenadas de los puntos de interés calculados
            cv::correctMatches(fundamental, p1, p2, np1, np2);
            int i = 0;
            for(auto it = outMatches.begin(); it != outMatches.end(); ++it){
                // KP de la izda
                kp1[it->queryIdx].pt = np1[i];
                // KP de la dcha
                kp2[it->trainIdx].pt = np2[i];
                i++;
            }
        }
        return fundamental;

    }
}

// Métodos públicos
namespace TFG{

    robustMatcher::robustMatcher(const cv::Ptr<cv::FeatureDetector> &detector_, const cv::Ptr<cv::DescriptorExtractor> &descriptor_) :
    detector(detector_), descriptor(descriptor_), normType(cv::NORM_HAMMING), ratio(0.8f),
    refineF(true), refineM(true), confidence(0.98), distance(1.0)
    {
        // Si no se ha introducido ningun extractor de descriptores como parametro se asigna a este el detector
        if(!this->descriptor){
            this->descriptor = this -> detector;
        }
    }

    cv::Mat robustMatcher::match(  const cv::Mat& im1, const cv::Mat& im2, const cv::Mat& mask1, const cv::Mat& mask2,
                    std::vector<cv::DMatch>& matches, std::vector<cv::KeyPoint>& kp1_filtrado,
                    std::vector<cv::KeyPoint>& kp2_filtrado, std::vector<cv::Point2f>& p1,
                    std::vector<cv::Point2f>& p2){

        std::vector<cv::KeyPoint> kp1, kp2;

        // 1. Detectar KP
        detector->detect(im1, kp1);
        detector->detect(im2, kp2);

        // 2. Filtrar KP
        cv::Mat mask = mask1 & im1;
        cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
        for(int i = 0; i < kp1.size(); i++){
            int fila = (int) kp1[i].pt.y;
            int columna = (int) kp1[i].pt.x;
            uint8_t valor_mascara = mask.data[fila*mask.step +  columna];
            // Si cae fuera de los objetos
            if(valor_mascara == 0){
                kp1_filtrado.push_back(kp1[i]);
            }
        }


        mask = mask2 & im2;
        cv::cvtColor(mask, mask, cv::COLOR_BGR2GRAY);
        for(int i = 0; i < kp2.size(); i++){
            int fila = (int) kp2[i].pt.y;
            int columna = (int) kp2[i].pt.x;
            uint8_t valor_mascara = mask.data[fila*mask.step +  columna];
            // Si cae fuera de los objetos
            if(valor_mascara == 0){
                kp2_filtrado.push_back(kp2[i]);
            }
        }

        // 3. Computar descriptores
        cv::Mat d1, d2, d1_filtrado, d2_filtrado;
        descriptor->compute(im1,kp1_filtrado, d1);
        descriptor->compute(im2,kp2_filtrado, d2);



        // 4. Calcular coincidencias
        std::vector<cv::DMatch> outputMatches;
        cv::BFMatcher matcher(normType /* Distancia */, false /* CrossCheck */);
        std::vector<std::vector<cv::DMatch>> vecino;
        matcher.knnMatch(d1, d2, vecino, 2);
        for (int i = 0; i < vecino.size(); i++) {
            if (vecino[i][0].distance < ratio * vecino[i][1].distance) {
                outputMatches.push_back(vecino[i][0]);
            }
        }
        // 5. Validar matches usando RANSAC
        cv::Mat fundamental = ransacTest(outputMatches, kp1_filtrado, kp2_filtrado, matches,p1, p2);

        return fundamental;

    }


    cv::Mat robustMatcher::match(   const cv::Mat& im1, const cv::Mat& im2, std::vector<cv::DMatch>& matches,
                                    std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
                                    std::vector<cv::Point2f>& p1, std::vector<cv::Point2f>& p2){
        // 1. Detectar KP
        detector->detect(im1, kp1);
        detector->detect(im2, kp2);
        cv::Mat test1;

        // 2. Computar descriptores
        cv::Mat d1, d2;
        descriptor->compute(im1,kp1, d1);
        descriptor->compute(im2,kp2, d2);


        // 3. Calcular coincidencias
        std::vector<cv::DMatch> outputMatches;
        cv::BFMatcher matcher(normType /* Distancia */, false /* CrossCheck */);
        std::vector<std::vector<cv::DMatch>> vecino;
        matcher.knnMatch(d1, d2, vecino, 2);
        for (int i = 0; i < vecino.size(); i++) {
            if (vecino[i][0].distance < ratio * vecino[i][1].distance) {
                outputMatches.push_back(vecino[i][0]);
            }
        }

        // 4. Validar matches usando RANSAC
        cv::Mat fundamental = ransacTest(outputMatches, kp1, kp2, matches,p1, p2);

        return fundamental;
    }

    void robustMatcher::simpleMatch(       const cv::Mat& im1,
                            const cv::Mat& im2,
                            std::vector<cv::DMatch>& matches,
                            std::vector<cv::KeyPoint>& kp1,
                            std::vector<cv::KeyPoint>& kp2      ){
        matches.clear();
        kp1.clear();
        kp2.clear();
        // 1. Detectar KP
        detector->detect(im1, kp1);
        detector->detect(im2, kp2);
        cv::Mat test1;

        // 2. Computar descriptores
        cv::Mat d1, d2;
        descriptor->compute(im1,kp1, d1);
        descriptor->compute(im2,kp2, d2);


        // 3. Calcular coincidencias
        cv::BFMatcher matcher(normType /* Distancia */, false /* CrossCheck */);
        std::vector<std::vector<cv::DMatch>> vecino;
        matcher.knnMatch(d1, d2, vecino, 2);
        for (int i = 0; i < vecino.size(); i++) {
            if (vecino[i][0].distance < 0.9 * vecino[i][1].distance) {
                matches.push_back(vecino[i][0]);
            }
        }
/*
        cv::Mat out;
        cv::drawMatches(im1, kp1, im2, kp2, matches, out);
        cv::imshow("matches", out);
        cv::waitKey();
*/
    }

}
