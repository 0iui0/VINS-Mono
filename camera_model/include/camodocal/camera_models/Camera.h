#ifndef CAMERA_H
#define CAMERA_H

#include <boost/shared_ptr.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <vector>

#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>

#define CV_BGR2GRAY cv::COLOR_BGRA2GRAY
#define CV_GRAY2BGR cv::COLOR_GRAY2BGR
#define CV_GRAY2RGB cv::COLOR_GRAY2RGB
#define CV_CALIB_CB_ADAPTIVE_THRESH cv::CALIB_CB_ADAPTIVE_THRESH
#define CV_CALIB_CB_NORMALIZE_IMAGE cv::CALIB_CB_NORMALIZE_IMAGE
#define CV_CALIB_CB_FILTER_QUADS cv::CALIB_CB_FILTER_QUADS
#define CV_CALIB_CB_FAST_CHECK cv::CALIB_CB_FAST_CHECK
#define CV_ADAPTIVE_THRESH_MEAN_C cv::ADAPTIVE_THRESH_MEAN_C
#define CV_THRESH_BINARY cv::THRESH_BINARY
#define CV_SHAPE_CROSS cv::MORPH_CROSS
#define CV_RETR_CCOMP cv::RETR_CCOMP
#define CV_CHAIN_APPROX_SIMPLE cv::CHAIN_APPROX_SIMPLE
#define CV_THRESH_BINARY_INV cv::THRESH_BINARY_INV

namespace camodocal {

    class Camera {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        enum ModelType {
            KANNALA_BRANDT,
            MEI,
            PINHOLE,
            SCARAMUZZA
        };

        class Parameters {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            Parameters(ModelType modelType);

            Parameters(ModelType modelType, const std::string &cameraName,
                       int w, int h);

            ModelType &modelType(void);

            std::string &cameraName(void);

            int &imageWidth(void);

            int &imageHeight(void);

            ModelType modelType(void) const;

            const std::string &cameraName(void) const;

            int imageWidth(void) const;

            int imageHeight(void) const;

            int nIntrinsics(void) const;

            virtual bool readFromYamlFile(const std::string &filename) = 0;

            virtual void writeToYamlFile(const std::string &filename) const = 0;

        protected:
            ModelType m_modelType;
            int m_nIntrinsics;
            std::string m_cameraName;
            int m_imageWidth;
            int m_imageHeight;
        };

        virtual ModelType modelType(void) const = 0;

        virtual const std::string &cameraName(void) const = 0;

        virtual int imageWidth(void) const = 0;

        virtual int imageHeight(void) const = 0;

        virtual cv::Mat &mask(void);

        virtual const cv::Mat &mask(void) const;

        virtual void estimateIntrinsics(const cv::Size &boardSize,
                                        const std::vector <std::vector<cv::Point3f>> &objectPoints,
                                        const std::vector <std::vector<cv::Point2f>> &imagePoints) = 0;

        virtual void estimateExtrinsics(const std::vector <cv::Point3f> &objectPoints,
                                        const std::vector <cv::Point2f> &imagePoints,
                                        cv::Mat &rvec, cv::Mat &tvec) const;

        // Lift points from the image plane to the sphere
        virtual void liftSphere(const Eigen::Vector2d &p, Eigen::Vector3d &P) const = 0;
        //%output P

        // Lift points from the image plane to the projective space
        virtual void liftProjective(const Eigen::Vector2d &p, Eigen::Vector3d &P) const = 0;
        //%output P

        // Projects 3D points to the image plane (Pi function)
        virtual void spaceToPlane(const Eigen::Vector3d &P, Eigen::Vector2d &p) const = 0;
        //%output p

        // Projects 3D points to the image plane (Pi function)
        // and calculates jacobian
        //virtual void spaceToPlane(const Eigen::Vector3d& P, Eigen::Vector2d& p,
        //                          Eigen::Matrix<double,2,3>& J) const = 0;
        //%output p
        //%output J

        virtual void undistToPlane(const Eigen::Vector2d &p_u, Eigen::Vector2d &p) const = 0;
        //%output p

        //virtual void initUndistortMap(cv::Mat& map1, cv::Mat& map2, double fScale = 1.0) const = 0;
        virtual cv::Mat initUndistortRectifyMap(cv::Mat &map1, cv::Mat &map2,
                                                float fx = -1.0f, float fy = -1.0f,
                                                cv::Size imageSize = cv::Size(0, 0),
                                                float cx = -1.0f, float cy = -1.0f,
                                                cv::Mat rmat = cv::Mat::eye(3, 3, CV_32F)) const = 0;

        virtual int parameterCount(void) const = 0;

        virtual void readParameters(const std::vector<double> &parameters) = 0;

        virtual void writeParameters(std::vector<double> &parameters) const = 0;

        virtual void writeParametersToYamlFile(const std::string &filename) const = 0;

        virtual std::string parametersToString(void) const = 0;

        /**
         * \brief Calculates the reprojection distance between points
         *
         * \param P1 first 3D point coordinates
         * \param P2 second 3D point coordinates
         * \return euclidean distance in the plane
         */
        double reprojectionDist(const Eigen::Vector3d &P1, const Eigen::Vector3d &P2) const;

        double reprojectionError(const std::vector <std::vector<cv::Point3f>> &objectPoints,
                                 const std::vector <std::vector<cv::Point2f>> &imagePoints,
                                 const std::vector <cv::Mat> &rvecs,
                                 const std::vector <cv::Mat> &tvecs,
                                 cv::OutputArray perViewErrors = cv::noArray()) const;

        double reprojectionError(const Eigen::Vector3d &P,
                                 const Eigen::Quaterniond &camera_q,
                                 const Eigen::Vector3d &camera_t,
                                 const Eigen::Vector2d &observed_p) const;

        void projectPoints(const std::vector <cv::Point3f> &objectPoints,
                           const cv::Mat &rvec,
                           const cv::Mat &tvec,
                           std::vector <cv::Point2f> &imagePoints) const;

    protected:
        cv::Mat m_mask;
    };

    typedef boost::shared_ptr <Camera> CameraPtr;
    typedef boost::shared_ptr<const Camera> CameraConstPtr;

}

#endif
