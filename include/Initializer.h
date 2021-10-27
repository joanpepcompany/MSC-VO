/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>

#include "Frame.h"
#include "auxiliar.h"


namespace ORB_SLAM2
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer
{
    typedef pair<int,int> Match;

public:

    // Fix the reference frame
    Initializer(const Frame &ReferenceFrame, float sigma = 1.0, int iterations = 200);

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated);

    bool Initialize(const Frame &CurrentFrame, const vector<int> &vMatches12,
                    cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated,
                    vector<int> &vLineMatches12, vector<cv::Point3f> &vLineS3D, vector<cv::Point3f> &vLineE3D,
                    vector<bool> &vbLineTriangulated);

    cv::Mat cubemapMatch_rgb;

private:

    void FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21);
    void FindFundamental(vector<bool> &vbInliers, float &score, cv::Mat &F21);

    cv::Mat ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);
    cv::Mat ComputeF21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2);

    float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &vbMatchesInliers, float sigma);

    float CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma);

    bool ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    bool ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

    void Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

    void LineTriangulate(const KeyLine &kl1, const KeyLine &kl2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &LineStart3D, cv::Mat &LineEnd3D);
    void LineTriangulate(const KeyLine &kl1, const KeyLine &kl2, const cv::Mat &P1, const cv::Mat &P2, const Vector3d &klf1, const Vector3d &klf2,
                         cv::Mat &LineStart3D, cv::Mat &LineEnd3D);

    void ReconstructLine(vector<Match> &vLineMatchesH, cv::Mat &K, cv::Mat &R21, cv::Mat &t21, vector<Point3f> &vLineS3D, vector<Point3f> &vLineE3D, vector<bool> &vinliers);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    ////////////////////////////////////////////////////////////////////////////////////
    // Lines
    vector<KeyLine> mvKeyLines1;
    vector<Vector3d> mvKeyLineFunctions1;
    vector<KeyLine> mvKeyLines2;
    vector<Vector3d> mvKeyLineFunctions2;
    vector<Match> mvLineMatches12;
    vector<bool> mvbLineMatched1;

    ////////////////////////////////////////////////////////////////////////////////////
    // Points
    // Keypoints from Frame 1, Frame 2
    vector<cv::KeyPoint> mvKeys1;
    vector<cv::KeyPoint> mvKeys2;
    vector<Match> mvMatches12;
    vector<bool> mvbMatched1; 

    // Calibration
    cv::Mat mK;     
    // Standard Deviation and Variance
    float mSigma, mSigma2; 

    // Ransac max iterations
    int mMaxIterations;    

    // Ransac sets
    vector<vector<size_t> > mvSets;  

};

} //namespace ORB_SLAM

#endif // INITIALIZER_H
