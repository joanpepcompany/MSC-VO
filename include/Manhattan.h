
#ifndef MANHATTAN_H
#define MANHATTAN_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <chrono>
#include "MapLine.h"
#include "Frame.h"

#include <mutex>
class MapLine;
class Frame;

namespace ORB_SLAM2
{
    class Manhattan
    {
    public:
        Manhattan();
        Manhattan(const cv::Mat &K);

        void computeStructConstInMap(Frame &frame, const vector<MapLine *> &mapLines);

        void computeStructConstrains(Frame &pF, MapLine * pML, std::vector<int> &v_idx_par, std::vector<int> &v_idx_perp);
        
        void computeStructConstrains(Frame &pF, const int &idx, std::vector<int> &idxPar, std::vector<int> &idxPerp);

        cv::Mat rotCW(const cv::Mat &line_eq, const cv::Mat rotation);

        void computeNormalsLPVO(const cv::Mat &im_depth_resized, const cv::Mat &K,
                                std::vector<cv::Mat> &pt_normals, std::vector<float> &depth_normals);
        void operator()(const cv::Mat &im_depth_resized, const cv::Mat &K,
                        std::vector<cv::Mat> &pt_normals,
                        std::vector<std::vector<cv::Mat>> &main_orient);

        void getMainOrientations(const std::vector<cv::Mat> &vec_normals,
                                 const int &min_samples,
                                 std::vector<cv::Mat> &selected_vectors,
                                 std::vector<std::vector<cv::Mat>> &represent_vect);

        bool extractCoarseManhAxes(const std::vector<cv::Mat> &v_normals,
                               const std::vector<cv::Mat> &cand_coord,
                               cv::Mat &manhattan_axis, float &succ_rate);

        void findCoordAxis(const std::vector<std::vector<cv::Mat>> &rep_lines,
                           std::vector<cv::Mat> &v_coord_axis);

        void LineManhAxisCorresp(const cv::Mat manh_axis,
                                      const std::vector<cv::Vec3f> &v_normals,
                                      std::vector<int> &line_axis);

        double computeAngle(const cv::Mat &vector1, const cv::Mat &vector2);

    private:
        void removeMatCol(cv::Mat &matIn, int col);
        void removeMatRow(cv::Mat &matIn, int row);

        void projectManhAxis(const cv::Mat &R_Mc,
                             const std::vector<cv::Mat> &normal_vector,
                             const double &sin_half_apex_angle,
                             std::vector<cv::Mat> &m_j);

        void MeanShift(const std::vector<cv::Mat> &F, const int &c,
                       cv::Mat &m, double &density);

        void RemoveRedundancyMF2(const std::vector<cv::Mat> &MF,
                                 std::vector<cv::Mat> &MF_can);

        void clusterMMF(const std::vector<cv::Mat> &MF_cann,
                        const float &ratio_th,
                        std::vector<cv::Mat> &MF_nonRd,
                        std::vector<float> &succ_rate);

        void EasyHist(const cv::Mat &data, const int &first,
                      const float &step, const int &last,
                      std::vector<std::vector<int>> &v_bin);


        double computeAngle2D(const cv::Mat &vector1, const cv::Mat &vector2);

        cv::Mat GetRandomRotation();
        cv::Mat euler2rot(const cv::Mat &euler);
        
        //  Member Variables 
        cv::Mat mK;
        float mFx;
        float mFy;
        float mCx;
        float mCy;
        float mInvFx;
        float mInvFy;

        double mCosThPar;
        double mCosThPerp;
        double mThManh2AxisAngle;

        std::vector<std::vector<int>> mvAxis;
        std::vector<cv::Mat> mRotPoss;
    };
}

#endif // MANHATTAN_H