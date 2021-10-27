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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>
#include "LocalMapping.h"
#include "lineIterator.h"
#include <unordered_set>

#define USE_CV_RECT

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight), mpManh(frame.mpManh),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
     mnScaleLevelsLine(frame.mnScaleLevelsLine),
     mfScaleFactorLine(frame.mfScaleFactorLine), mfLogScaleFactorLine(frame.mfLogScaleFactorLine),
     mvScaleFactorsLine(frame.mvScaleFactorsLine), mvInvScaleFactorsLine(frame.mvInvScaleFactorsLine),
     mvLevelSigma2Line(frame.mvLevelSigma2Line), mvInvLevelSigma2Line(frame.mvInvLevelSigma2Line),
     mLdesc(frame.mLdesc), NL(frame.NL), mvKeylinesUn(frame.mvKeylinesUn),mvSupposedVectors(frame.mvSupposedVectors), vManhAxisIdx(frame.vManhAxisIdx), mvPerpLines(frame.mvPerpLines), mvParallelLines(frame.mvParallelLines), mvpMapLines(frame.mvpMapLines), mvLines3D(frame.mvLines3D), 
     mvLineEq(frame.mvLineEq),
     mvbLineOutlier(frame.mvbLineOutlier), mvKeyLineFunctions(frame.mvKeyLineFunctions), ImageGray(frame.ImageGray.clone())
{
    // Points
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    // Lines
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGridForLine[i][j]=frame.mGridForLine[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

/// Stereo Frame
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, Manhattan* manh, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc), mpManh(manh), mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();
  
   // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();
    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

/// RGB-D Frame
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor, LINEextractor* lsdextractor, ORBVocabulary* voc, Manhattan* manh, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const cv::Mat &mask, const bool &bManhInit)
    :mpORBvocabulary(voc), mpManh(manh), mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),mpLSDextractorLeft(lsdextractor), 
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    imGray.copyTo(ImageGray);

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // Scale Level Info for line
    mnScaleLevelsLine = mpLSDextractorLeft->GetLevels();
    mfScaleFactorLine = mpLSDextractorLeft->GetScaleFactor();
    mfLogScaleFactorLine = log(mfScaleFactor);
    mvScaleFactorsLine = mpLSDextractorLeft->GetScaleFactors();
    mvInvScaleFactorsLine = mpLSDextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2Line = mpLSDextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2Line = mpLSDextractorLeft->GetInverseScaleSigmaSquares();
  
    // This is done only for the first Frame (or after a change in the calibration)
    if (mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
        mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        mbInitialComputations = false;
    }

     mb = mbf / fx;

     std::chrono::steady_clock::time_point t1_fet_extr = std::chrono::steady_clock::now();

     if (bManhInit)
     {
         thread threadPoint(&Frame::ExtractORBNDepth, this, imGray, imDepth);
         thread threadLine(&Frame::ExtractLSD, this, imGray, imDepth);
         threadPoint.join();
         threadLine.join();
     }
    // Extract pt normals until the Manh. Axes is computed
     else
     {
         thread threadPoint(&Frame::ExtractORBNDepth, this, imGray, imDepth);
         thread threadLine(&Frame::ExtractLSD, this, imGray, imDepth);
         thread threadNormals(&Frame::ExtractMainImgPtNormals, this, imDepth, K);
         threadPoint.join();
         threadLine.join();
         threadNormals.join();
     }

    
    std::chrono::steady_clock::time_point t2_fet_extr = std::chrono::steady_clock::now();
    chrono::duration<double> time_fet_extr = chrono::duration_cast<chrono::duration<double>>(t2_fet_extr - t1_fet_extr);
    MTimeFeatExtract = time_fet_extr.count();

    NL = mvKeylinesUn.size();

    if (mvKeys.empty())
        return;

    if (!bManhInit)
    {
        // TODO 0: Avoid this conversion
        std::vector<cv::Mat> v_line_eq;

        for (size_t i = 0; i < mvLineEq.size(); i++)
        {
            if (mvLineEq[i][2] == -1 || mvLineEq[i][2] == 0)
                continue;

            cv::Mat line_vector = (Mat_<double>(3, 1) << mvLineEq[i][0],
                                   mvLineEq[i][1],
                                   mvLineEq[i][2]);
            std::vector<cv::Mat> v_line;                                   
            v_line.push_back(line_vector);
            mRepLines.push_back(v_line);
        }
    }

     mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
     mvbOutlier = vector<bool>(N, false);

     mvpMapLines = vector<MapLine *>(NL, static_cast<MapLine *>(NULL));
     mvbLineOutlier = vector<bool>(NL, false);

     thread threadAssignPoint(&Frame::AssignFeaturesToGrid, this);
     thread threadAssignLine(&Frame::AssignFeaturesToGridForLine, this);
     threadAssignPoint.join();
     threadAssignLine.join();

     mvParallelLines = vector<std::vector<MapLine *>*>(NL, static_cast<std::vector<MapLine *>*>(nullptr));
     mvPerpLines = vector<std::vector<MapLine *>*>(NL, static_cast<std::vector<MapLine *>*>(nullptr));

    mvParLinesIdx = std::vector<std::vector<int>*>(NL, static_cast<std::vector<int>*>(nullptr));
    mvPerpLinesIdx = std::vector<std::vector<int>*>(NL, static_cast<std::vector<int>*>(nullptr));
}

/// RGB Frame
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor *orbextractor, LINEextractor *lsdextractor, ORBVocabulary *voc, Manhattan* manh, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const cv::Mat &mask)
    : mpORBvocabulary(voc), mpManh(manh), mpORBextractorLeft(orbextractor), mpORBextractorRight(static_cast<ORBextractor *>(NULL)), mpLSDextractorLeft(lsdextractor),
      mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    imGray.copyTo(ImageGray);

    // Scale Level Info for point
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // Scale Level Info for line
    mnScaleLevelsLine = mpLSDextractorLeft->GetLevels();
    mfScaleFactorLine = mpLSDextractorLeft->GetScaleFactor();
    mfLogScaleFactorLine = log(mfScaleFactor);
    mvScaleFactorsLine = mpLSDextractorLeft->GetScaleFactors();
    mvInvScaleFactorsLine = mpLSDextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2Line = mpLSDextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2Line = mpLSDextractorLeft->GetInverseScaleSigmaSquares();

    cv::Mat mUndistX, mUndistY, mImGray_remap;
    initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3,3), mK, Size(imGray.cols, imGray.rows), CV_32F, mUndistX, mUndistY);
    cv::remap(imGray, mImGray_remap, mUndistX, mUndistY, cv::INTER_LINEAR);

    thread threadPoint(&Frame::ExtractORB, this, 0, imGray);
    thread threadLine(&Frame::ExtractLSD, this, mImGray_remap, mask);
    threadPoint.join();
    threadLine.join();

    NL = mvKeylinesUn.size(); 
    N = mvKeys.size();

    if(mvKeys.empty())
        return;
        
    //mvKeysUn = mvKeys;
    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));  
    mvbOutlier = vector<bool>(N,false);   

    mvpMapLines = vector<MapLine*>(NL,static_cast<MapLine*>(NULL));
    mvbLineOutlier = vector<bool>(NL,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    thread threadAssignPoint(&Frame::AssignFeaturesToGrid, this);
    thread threadAssignLine(&Frame::AssignFeaturesToGridForLine, this);
    threadAssignPoint.join();
    threadAssignLine.join();

}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::AssignFeaturesToGridForLine()
{
    int nReserve = 0.5f*NL/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGridForLine[i][j].reserve(nReserve);

    //#pragma omp parallel for
    for(int i=0;i<NL;i++)
    {
        const KeyLine &kl = mvKeylinesUn[i];

        list<pair<int, int>> line_coords;

        LineIterator* it = new LineIterator(kl.startPointX * mfGridElementWidthInv, kl.startPointY * mfGridElementHeightInv, kl.endPointX * mfGridElementWidthInv, kl.endPointY * mfGridElementHeightInv);

        std::pair<int, int> p;
        while (it->getNext(p))
            if (p.first >= 0 && p.first < FRAME_GRID_COLS && p.second >= 0 && p.second < FRAME_GRID_ROWS)
                mGridForLine[p.first][p.second].push_back(i);

        delete [] it;
    }
}

void Frame::ExtractORBNDepth(const cv::Mat &im, const cv::Mat &im_depth)
{
    (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);

    N = mvKeys.size();
    if (!mvKeys.empty())
    {
     UndistortKeyPoints();
    ComputeStereoFromRGBD(im_depth);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im )
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}


void Frame::ExtractLSD(const cv::Mat &im, const cv::Mat &im_depth)
{
     std::chrono::steady_clock::time_point t1_line_detect = std::chrono::steady_clock::now();

    cv::Mat mask;
    (*mpLSDextractorLeft)(im,mask,mvKeylinesUn, mLdesc, mvKeyLineFunctions);

     std::chrono::steady_clock::time_point t2_line_detect = std::chrono::steady_clock::now();
     chrono::duration<double> t_line_detect = chrono::duration_cast<chrono::duration<double>>(t2_line_detect - t1_line_detect);

     std::chrono::steady_clock::time_point t1_line_good = std::chrono::steady_clock::now();
    
    // Option 1: git yanyan-li/PlanarSLAM
     isLineGood(im, im_depth, mK);

    // Option 2: Changes proposed by us
    //  ComputeStereoFromRGBDLines(im_depth);
    
    // Option 3 Single Backprojection procedure
    //  ComputeDepthLines(im_depth);

     std::chrono::steady_clock::time_point t2_line_good = std::chrono::steady_clock::now();
     chrono::duration<double> time_line_good = chrono::duration_cast<chrono::duration<double>>(t2_line_good - t1_line_good);
}

// Optimize Lines --> Small mods of the code from YanYan Li ICRA 2021
void Frame::isLineGood(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &K)
{
    mvLineEq.clear();
    mvLineEq.resize(mvKeylinesUn.size(),Vec3f(-1.0, -1.0, -1.0));
    mvLines3D.resize(mvKeylinesUn.size(), std::make_pair(Vector3d(0.0, 0.0, 0.0), Vector3d(0.0, 0.0, 0.0)));

    for (int i = 0; i < mvKeylinesUn.size(); ++i)
    { // each line
        double len = cv::norm(mvKeylinesUn[i].getStartPoint() - mvKeylinesUn[i].getEndPoint());
        vector<cv::Point3d> pts3d;
        // iterate through a line
        double numSmp = (double)min((int)len, 20); //number of line points sampled

        pts3d.reserve(numSmp);
        for (int j = 0; j <= numSmp; ++j)
        {
            // use nearest neighbor to querry depth value
            // assuming position (0,0) is the top-left corner of image, then the
            // top-left pixel's center would be (0.5,0.5)
            cv::Point2d pt = mvKeylinesUn[i].getStartPoint() * (1 - j / numSmp) +
                             mvKeylinesUn[i].getEndPoint() * (j / numSmp);

            if (pt.x < 0 || pt.y < 0 || pt.x >= imDepth.cols || pt.y >= imDepth.rows)
            {
                continue;
            }
            int row, col; // nearest pixel for pt
            if ((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y))
            { // boundary issue
                col = max(int(pt.x - 1), 0);
                row = max(int(pt.y - 1), 0);
            }
            else
            {
                col = int(pt.x);
                row = int(pt.y);
            }

            float d = -1;
            if (imDepth.at<float>(row, col) <= 0.01)
            { 
                continue;
            }
            else
            {
                d = imDepth.at<float>(row, col);
            }
            cv::Point3d p;

            p.z = d;
            p.x = (col - cx) * p.z * invfx;
            p.y = (row - cy) * p.z * invfy;

            pts3d.push_back(p);
        }

        if (pts3d.size() < 5){
            continue;
        }

        RandomLine3d tmpLine;
        vector<RandomPoint3d> rndpts3d;
        rndpts3d.reserve(pts3d.size());

        // compute uncertainty of 3d points
        for (int j = 0; j < pts3d.size(); ++j)
        {
            rndpts3d.push_back(mpLSDextractorLeft->compPt3dCov(pts3d[j], K, 1));
        }
        // using ransac to extract a 3d line from 3d pts
        tmpLine = mpLSDextractorLeft->extract3dline_mahdist(rndpts3d);

        if (
        cv::norm(tmpLine.A - tmpLine.B) > 0.02)
        {

            Eigen::Vector3d st_pt3D(tmpLine.A.x, tmpLine.A.y, tmpLine.A.z);
            Eigen::Vector3d e_pt3D(tmpLine.B.x, tmpLine.B.y, tmpLine.B.z);

            cv::Vec3f line_eq(tmpLine.B.x - tmpLine.A.x, tmpLine.B.y- tmpLine.A.y, tmpLine.B.z - tmpLine.A.z); 

            float magn  = sqrt(line_eq[0] * line_eq[0] + line_eq[1] * line_eq[1]+ line_eq[2] * line_eq[2]);
            std::pair<Eigen::Vector3d, Eigen::Vector3d> line_ep_3D(st_pt3D, e_pt3D);
            mvLines3D[i] = line_ep_3D;

            mvLineEq[i] = line_eq/magn;
        }
    }
}

void Frame::ExtractMainImgPtNormals(const cv::Mat &img, const cv::Mat &K)
{
    // mRepNormals --> used to extract initial candidates of the Manh. Axes in the Coarse Manh. Estimation
    // mvPtNormals --> used to cooroborate and to refine the coarse Manh. Estimation.
     (*mpManh)(img,K,mvPtNormals, mRepNormals);
}

void Frame::lineDescriptorMAD( vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const
{
    vector<vector<DMatch>> matches_nn, matches_12;
    matches_nn = line_matches;
    matches_12 = line_matches;

    // estimate the NN's distance standard deviation
    double nn_dist_median;
    sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_dist_median = matches_nn[int(matches_nn.size()/2)][0].distance;

    for(unsigned int i=0; i<matches_nn.size(); i++)
        matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
    sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
    nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

    // estimate the NN's 12 distance standard deviation
    double nn12_dist_median;
    sort( matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
    nn12_dist_median = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
    for (unsigned int j=0; j<matches_12.size(); j++)
        matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
    sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
    nn12_mad = 1.4826 * matches_12[int(matches_12.size()/2)][0].distance;
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit)
{
    pML->mbTrackInView = false;

    Vector6d P = pML->GetWorldPos();

    cv::Mat SP = (Mat_<float>(3,1) << P(0), P(1), P(2));
    cv::Mat EP = (Mat_<float>(3,1) << P(3), P(4), P(5));

    const cv::Mat SPc = mRcw*SP + mtcw;
    const float &SPcX = SPc.at<float>(0);
    const float &SPcY = SPc.at<float>(1);
    const float &SPcZ = SPc.at<float>(2);

    const cv::Mat EPc = mRcw*EP + mtcw;
    const float &EPcX = EPc.at<float>(0);
    const float &EPcY = EPc.at<float>(1);
    const float &EPcZ = EPc.at<float>(2);

    if(SPcZ<0.0f || EPcZ<0.0f)
        return false;

    const float invz1 = 1.0f/SPcZ;
    const float u1 = fx * SPcX * invz1 + cx;
    const float v1 = fy * SPcY * invz1 + cy;

    if(u1<mnMinX || u1>mnMaxX)
        return false;
    if(v1<mnMinY || v1>mnMaxY)
        return false;

    const float invz2 = 1.0f/EPcZ;
    const float u2 = fx*EPcX*invz2 + cx;
    const float v2 = fy*EPcY*invz2 + cy;

    if(u2<mnMinX || u2>mnMaxX)
        return false;
    if(v2<mnMinY || v2>mnMaxY)
        return false;

    const float maxDistance = pML->GetMaxDistanceInvariance();
    const float minDistance = pML->GetMinDistanceInvariance();
 
    const cv::Mat OM = 0.5*(SP+EP) - mOw;
    const float dist = cv::norm(OM);

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Vector3d Pn = pML->GetNormal();
    cv::Mat pn = (Mat_<float>(3,1) << Pn(0), Pn(1), Pn(2));
    const float viewCos = OM.dot(pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);

    // Data used by the tracking
    pML->mbTrackInView = true;
    pML->mTrackProjX1 = u1;
    pML->mTrackProjY1 = v1;
    pML->mTrackProjX2 = u2;
    pML->mTrackProjY2 = v2;
    pML->mnTrackScaleLevel = nPredictedLevel;
    pML->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

vector<size_t> Frame::GetFeaturesInAreaForLine(const float &x1, const float &y1, const float &x2, const float &y2, const float  &r, const int minLevel, const int maxLevel,const float TH) const
{
    vector<size_t> vIndices;
    vIndices.reserve(NL);
    unordered_set<size_t> vIndices_set;

    float x[3] = {x1, (x1+x2)/2.0, x2};
    float y[3] = {y1, (y1+y2)/2.0, y2}; 

    float delta1x = x1-x2;
    float delta1y = y1-y2;
    float norm_delta1 = sqrt(delta1x*delta1x + delta1y*delta1y);
    delta1x /= norm_delta1;
    delta1y /= norm_delta1;

    for(int i = 0; i<3;i++){
        const int nMinCellX = max(0,(int)floor((x[i]-mnMinX-r)*mfGridElementWidthInv));
        if(nMinCellX>=FRAME_GRID_COLS)
            continue;

        const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x[i]-mnMinX+r)*mfGridElementWidthInv));
        if(nMaxCellX<0)
            continue;

        const int nMinCellY = max(0,(int)floor((y[i]-mnMinY-r)*mfGridElementHeightInv));
        if(nMinCellY>=FRAME_GRID_ROWS)
            continue;

        const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y[i]-mnMinY+r)*mfGridElementHeightInv));
        if(nMaxCellY<0)
            continue;

        for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
            {
                const vector<size_t> vCell = mGridForLine[ix][iy];
                if(vCell.empty())
                    continue;

                for(size_t j=0, jend=vCell.size(); j<jend; j++)
                {
                    if(vIndices_set.find(vCell[j]) != vIndices_set.end())
                        continue;

                    const KeyLine &klUn = mvKeylinesUn[vCell[j]];

                    float delta2x = klUn.startPointX - klUn.endPointX;
                    float delta2y = klUn.startPointY - klUn.endPointY;
                    float norm_delta2 = sqrt(delta2x*delta2x + delta2y*delta2y);
                    delta2x /= norm_delta2;
                    delta2y /= norm_delta2;
                    float CosSita = abs(delta1x * delta2x + delta1y * delta2y);

                    if(CosSita < TH)
                        continue;

                    Eigen::Vector3d Lfunc = mvKeyLineFunctions[vCell[j]]; 
                    const float dist = Lfunc(0)*x[i] + Lfunc(1)*y[i] + Lfunc(2);

                    if(fabs(dist)<r)
                    {
                        if(vIndices_set.find(vCell[j]) == vIndices_set.end())
                        {
                            vIndices.push_back(vCell[j]);
                            vIndices_set.insert(vCell[j]);
                        }
                    }
                }
            }
        }
    }
    
    return vIndices;
}

vector<size_t> Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                                     const int minLevel, const int maxLevel, const float TH) const
{
    vector<size_t> vIndices;

    vector<KeyLine> vkl = this->mvKeylinesUn;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>0);

    float delta1x = x1-x2;
    float delta1y = y1-y2;
    float norm_delta1 = sqrt(delta1x*delta1x + delta1y*delta1y);
    delta1x /= norm_delta1;
    delta1y /= norm_delta1;

    for(size_t i=0; i<vkl.size(); i++)
    {
        KeyLine keyline = vkl[i];

        float distance = (0.5*(x1+x2)-keyline.pt.x)*(0.5*(x1+x2)-keyline.pt.x)+(0.5*(y1+y2)-keyline.pt.y)*(0.5*(y1+y2)-keyline.pt.y);
        if(distance > r*r)
            continue;

        float delta2x = vkl[i].startPointX - vkl[i].endPointX;
        float delta2y = vkl[i].startPointY - vkl[i].endPointY;
        float norm_delta2 = sqrt(delta2x*delta2x + delta2y*delta2y);
        delta2x /= norm_delta2;
        delta2y /= norm_delta2;
        float CosSita = abs(delta1x * delta2x + delta1y * delta2y);

        if(CosSita < TH)
            continue;

        if(bCheckLevels)
        {
            if(keyline.octave<minLevel)
                continue;
            if(maxLevel>=0 && keyline.octave>maxLevel)
                continue;
        }

        vIndices.push_back(i);
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);   
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N); 
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0 && d < 7.0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

void Frame::ComputeDepthLines(const cv::Mat imDepth)
{
    mvLines3D.clear();
    mvLines3D.resize(mvKeylinesUn.size(), std::make_pair(Vector3d(0.0, 0.0, 0.0),Vector3d(0.0, 0.0, 0.0)));

    mvLineEq.clear();
    mvLineEq.resize(mvKeylinesUn.size(),Vec3f(-1.0, -1.0, -1.0));

     for (int i = 0; i < mvKeylinesUn.size(); i++)
    {
         std::pair<cv::Point3f, cv::Point3f> pair_pts_3D = std::make_pair(cv::Point3f(-1.0, -1.0, -1.0),cv::Point3f(-1.0, -1.0, -1.0));

        if (!ComputeDepthEnpoints(imDepth, mvKeylinesUn[i], mK, pair_pts_3D))
         {
            mvKeylinesUn[i].startPointX = 0;
            mvKeylinesUn[i].startPointY = 0;
            mvKeylinesUn[i].endPointX = 0;
            mvKeylinesUn[i].endPointY = 0;
        }

        else
        {
            mvLineEq[i] = pair_pts_3D.second - pair_pts_3D.first;
         
            Eigen::Vector3d st_pt3D(pair_pts_3D.first.x,
                                    pair_pts_3D.first.y,
                                    pair_pts_3D.first.z);

            Eigen::Vector3d e_pt3D(pair_pts_3D.second.x,
                                   pair_pts_3D.second.y,
                                   pair_pts_3D.second.z);

            std::pair<Eigen::Vector3d, Eigen::Vector3d> line_ep_3D(st_pt3D, e_pt3D);
            mvLines3D[i] = line_ep_3D;
        }
    }
    

}

void Frame::ComputeStereoFromRGBDLines(const cv::Mat imDepth)
{
    mvLines3D.clear();
    mvLines3D.resize(mvKeylinesUn.size(), std::make_pair(Vector3d(0.0, 0.0, 0.0),Vector3d(0.0, 0.0, 0.0)));

    mvLineEq.clear();
    mvLineEq.resize(mvKeylinesUn.size(),Vec3f(-1.0, -1.0, -1.0));

    // #pragma omp parallel for
    for (int i = 0; i < mvKeylinesUn.size(); i++)
    {
        if (mvKeylinesUn[i].lineLength < 10.0)
        {
            mvKeylinesUn[i].startPointX = 0;
            mvKeylinesUn[i].startPointY = 0;
            mvKeylinesUn[i].endPointX = 0;
            mvKeylinesUn[i].endPointY = 0;
            continue;
        }

         std::pair<cv::Point, cv::Point> pair_pts_2D;
         std::pair<cv::Point3f, cv::Point3f> pair_pts_3D = std::make_pair(cv::Point3f(-1.0, -1.0, -1.0),cv::Point3f(-1.0, -1.0, -1.0));
         cv::Vec3f line_vector;

         if (!mpLSDextractorLeft->computeBest3dLineRepr(ImageGray, imDepth, mvKeylinesUn[i], mK, pair_pts_2D, pair_pts_3D, line_vector))
         {
            mvKeylinesUn[i].startPointX = 0;
            mvKeylinesUn[i].startPointY = 0;
            mvKeylinesUn[i].endPointX = 0;
            mvKeylinesUn[i].endPointY = 0;
        }

        else
        {
            mvLineEq[i] = line_vector;
         
            Eigen::Vector3d st_pt3D(pair_pts_3D.first.x,
                                    pair_pts_3D.first.y,
                                    pair_pts_3D.first.z);

            Eigen::Vector3d e_pt3D(pair_pts_3D.second.x,
                                   pair_pts_3D.second.y,
                                   pair_pts_3D.second.z);

            std::pair<Eigen::Vector3d, Eigen::Vector3d> line_ep_3D(st_pt3D, e_pt3D);
            mvLines3D[i] = line_ep_3D;
        }
    }
}

bool Frame::ComputeDepthEnpoints(const cv::Mat &imDepth, const line_descriptor::KeyLine &keyline, const cv::Mat mK, std::pair<cv::Point3f, cv::Point3f> &end_pts3D)
{
    cv::Point2f st_pt = keyline.getStartPoint(); 

    const float &st_v = st_pt.y;
    const float &st_u = st_pt.x;

    const float st_d = imDepth.at<float>(st_v, st_u);

    if (!(st_d > 0 && st_d < 7.0))
        return false;
    
    cv::Point2f end_pt = keyline.getEndPoint();
    
    const float &end_v = end_pt.y;
    const float &end_u = end_pt.x;

    const float end_d = imDepth.at<float>(end_v, end_u);

    if (!(end_d > 0 && end_d < 7.0))
        return false;  
    
    float st_x = ((st_pt.x - mK.at<float>(0, 2)) * st_d) / mK.at<float>(0, 0);
    float st_y = ((st_pt.y - mK.at<float>(1, 2)) * st_d) / mK.at<float>(1, 1);
    cv::Point3f st_point(st_x, st_y, st_d);

    float end_x = ((end_pt.x - mK.at<float>(0, 2)) * end_d) / mK.at<float>(0, 0);
    float end_y = ((end_pt.y - mK.at<float>(1, 2)) * end_d) / mK.at<float>(1, 1);
    cv::Point3f end_point(end_x, end_y, end_d);

    end_pts3D.first =  st_point;
    end_pts3D.second = end_point;
}


cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
