#ifndef LINEEXTRACTOR_H
#define LINEEXTRACTOR_H

#include <iostream>
#include <vector>
#include <list>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "auxiliar.h"

using namespace std;
using namespace cv;


namespace ORB_SLAM2{

    template <class bidiiter>
    //Fisher-Yates shuffle
    bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random)
    {
        size_t left = std::distance(begin, end);
        while (num_random--)
        {
            bidiiter r = begin;
            std::advance(r, rand() % left);
            std::swap(*begin, *r);
            ++begin;
            --left;
        }
        return begin;
    }

    typedef struct RandomPoint3ds {
    cv::Point3d pos;
    double xyz[3];
    double W_sqrt[3]; // used for mah-dist from pt to ln
    cv::Mat cov;
    cv::Mat U, W; // cov = U*D*U.t, D = diag(W); W is vector
    double DU[9];
    double dux[3];

    RandomPoint3ds() {}

    RandomPoint3ds(cv::Point3d _pos) {
        pos = _pos;
        xyz[0] = _pos.x;
        xyz[1] = _pos.y;
        xyz[2] = _pos.z;

        cov = cv::Mat::eye(3, 3, CV_64F);
        U = cv::Mat::eye(3, 3, CV_64F);
        W = cv::Mat::ones(3, 1, CV_64F);
    }

    RandomPoint3ds(cv::Point3d _pos, cv::Mat _cov) {
        pos = _pos;
        xyz[0] = _pos.x;
        xyz[1] = _pos.y;
        xyz[2] = _pos.z;
        cov = _cov.clone();
        cv::SVD svd(cov);
        U = svd.u.clone();
        W = svd.w.clone();
        W_sqrt[0] = sqrt(svd.w.at<double>(0));
        W_sqrt[1] = sqrt(svd.w.at<double>(1));
        W_sqrt[2] = sqrt(svd.w.at<double>(2));

        cv::Mat D = (cv::Mat_<double>(3, 3) << 1 / W_sqrt[0], 0, 0,
                0, 1 / W_sqrt[1], 0,
                0, 0, 1 / W_sqrt[2]);
        cv::Mat du = D * U.t();
        DU[0] = du.at<double>(0, 0);
        DU[1] = du.at<double>(0, 1);
        DU[2] = du.at<double>(0, 2);
        DU[3] = du.at<double>(1, 0);
        DU[4] = du.at<double>(1, 1);
        DU[5] = du.at<double>(1, 2);
        DU[6] = du.at<double>(2, 0);
        DU[7] = du.at<double>(2, 1);
        DU[8] = du.at<double>(2, 2);
        dux[0] = DU[0] * pos.x + DU[1] * pos.y + DU[2] * pos.z;
        dux[1] = DU[3] * pos.x + DU[4] * pos.y + DU[5] * pos.z;
        dux[2] = DU[6] * pos.x + DU[7] * pos.y + DU[8] * pos.z;

    }
} RandomPoint3d;

typedef struct RandomLine3ds {

    std::vector<RandomPoint3d> pts;  //supporting collinear points
    cv::Point3d A, B;
    cv::Point3d director; //director
    cv::Point3d direct1;
    cv::Point3d direct2;
    cv::Point3d mid;//middle point between two end points
    cv::Mat covA, covB;
    RandomPoint3d rndA, rndB;
    cv::Point3d u, d; // following the representation of Zhang's paper 'determining motion from...'
    RandomLine3ds() {}

    RandomLine3ds(cv::Point3d _A, cv::Point3d _B, cv::Mat _covA, cv::Mat _covB) {
        A = _A;
        B = _B;
        covA = _covA.clone();
        covB = _covB.clone();
    }

} RandomLine3d;

typedef struct FrameLines
// FrameLine represents a line segment detected from a rgb-d frame.
// It contains 2d image position (endpoints, line equation), and 3d info (if
// observable from depth image).
{

    cv::Point2d p, q;                // image endpoints p and q
    cv::Mat l;                    // 3-vector of image line equation,
    double lineEq2d[3];
    cv::Point3d direction;
    cv::Point3d direct1;
    cv::Point3d direct2;
    bool haveDepth;            // whether have depth
    //RandomLine3d line3d;
    std::vector<RandomPoint3d> rndpts3d;

    cv::Point2d r;                    // image line gradient direction (polarity);
    cv::Mat des;                // image line descriptor;

    int lid;                // local id in frame
    int gid;                // global id;

    int lid_prvKfrm;        // correspondence's lid in previous keyframe

    FrameLines() { gid = -1; }

    FrameLines(cv::Point2d p_, cv::Point2d q_);

    cv::Point2d getGradient(cv::Mat *xGradient, cv::Mat *yGradient);

    void complineEq2d()
    {
        cv::Mat pt1 = (cv::Mat_<double>(3, 1) << p.x, p.y, 1);
        cv::Mat pt2 = (cv::Mat_<double>(3, 1) << q.x, q.y, 1);
        cv::Mat lnEq = pt1.cross(pt2); // lnEq = pt1 x pt2
        lnEq = lnEq / sqrt(lnEq.at<double>(0) * lnEq.at<double>(0)
                           + lnEq.at<double>(1) * lnEq.at<double>(1)); // normalize, optional
        lineEq2d[0] = lnEq.at<double>(0);
        lineEq2d[1] = lnEq.at<double>(1);
        lineEq2d[2] = lnEq.at<double>(2);
    }
} FrameLine;

class LINEextractor
{
    
public:
    LINEextractor(){}
    LINEextractor( int _numOctaves, float _scale, unsigned int _nLSDFeature, double _min_line_length);
    ~LINEextractor(){}

    void operator()( cv::InputArray image, cv::InputArray mask, std::vector<line_descriptor::KeyLine>& keylines, cv::OutputArray descriptors, std::vector<Eigen::Vector3d> &lineVec2d);

    bool computeBest3dLineRepr(const cv::Mat &rgb_image, const cv::Mat &depth_img, const line_descriptor::KeyLine &keyline, const cv::Mat mK, std::pair<cv::Point, cv::Point> &pair_pts_2D, std::pair<cv::Point3f, cv::Point3f> &end_pts3D, cv::Vec3f &line_vector);

    float computeAngle(const cv::Point3f &pt1, const cv::Point3f &pt2, const cv::Point3f &pt3, const cv::Point3f &pt4);
    double computeAngle(const cv::Mat &vector1, const cv::Mat &vector2);
   
    // git yanyan-li/PlanarSLAM
    // -------
    RandomPoint3d compPt3dCov(cv::Point3d pt, cv::Mat K, double time_diff_sec);
    RandomLine3d extract3dline_mahdist(const vector<RandomPoint3d> &pts);
    double mah_dist3d_pt_line(const RandomPoint3d &pt, const cv::Point3d &q1, const cv::Point3d &q2);
    bool verify3dLine(const vector<RandomPoint3d> &pts, const cv::Point3d &A, const cv::Point3d &B);
    void computeLine3d_svd (const vector<RandomPoint3d>& pts, const vector<int>& idx, cv::Point3d& mean, cv::Point3d& drct);

    cv::Mat array2mat(double a[], int n) // inhomo mat
    // n is the size of a[]
    {
        return cv::Mat(n, 1, CV_64F, a);
    }

    cv::Point3d mat2cvpt3d(cv::Mat m)
    // 3x1 mat => point
    {
        if (m.cols * m.rows == 3)
            return cv::Point3d(m.at<double>(0),
                               m.at<double>(1),
                               m.at<double>(2));
        else
            cerr << "input matrix dimmension wrong!";
    }

    cv::Point3d projPt3d2Ln3d(const cv::Point3d &P, const cv::Point3d &mid, const cv::Point3d &drct)
    // project a 3d point P to a 3d line (represented with midpt and direction)
    {
        cv::Point3d A = mid;
        cv::Point3d B = mid + drct;
        cv::Point3d AB = B - A;
        cv::Point3d AP = P - A;
        return A + (AB.dot(AP) / (AB.dot(AB))) * AB;
    }
    //  -------


    int inline GetLevels()
    {
        return numOctaves;
    }

    float inline GetScaleFactor(){
        return scale;}

    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

private:
    void getPointsFromLineAndUnproject(const cv::Mat image, const cv::Mat depth_image, const cv::Mat mk, const line_descriptor::KeyLine &line, std::vector<cv::Point3f> &line_points);

    bool computeEndpoints(const std::vector<cv::Point3f> &threeD_points, std::pair<cv::Point3f, cv::Point3f> &end_pts, cv::Vec3f &line_vector);

    cv::Point3f projectionPt2Line(const cv::Point3f &point, const cv::Vec6f &vector_n_pt);

protected:
    double min_line_length;
    int numOctaves;
    unsigned int nLSDFeature;
    float scale;

    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

};
}

#endif