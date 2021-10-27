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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    int img_width = fSettings["Camera.width"];
    int img_height = fSettings["Camera.height"];

    if((mask = imread("./masks/mask.png", cv::IMREAD_GRAYSCALE)).empty())
        mask = cv::Mat();

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    int nFeaturesLine = fSettings["LINEextractor.nFeatures"];
    float fScaleFactorLine = fSettings["LINEextractor.scaleFactor"];
    int nLevelsLine = fSettings["LINEextractor.nLevels"];
    int min_length = fSettings["LINEextractor.min_line_length"];

    mpLSDextractorLeft = new LINEextractor(nLevelsLine, fScaleFactorLine, nFeaturesLine, min_length);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    // Initialization of global Manhattan variables     
    mpManh = new Manhattan(K);
    mManhInit = false;
    mCoarseManhInit = false;

    // Initialization of global time variables     
    mSumMTimeFeatExtract = 0.0;
    mSumMTimeEptsLineOpt = 0.0;
    mSumTimePoseEstim = 0.0;
    mTimeCoarseManh = 0.0;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mpManh,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);
    
    // Extract Frame
    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft, mpLSDextractorLeft, mpORBVocabulary,mpManh,mK,mDistCoef,mbf,mThDepth,mask, mCoarseManhInit);

    mSumMTimeFeatExtract +=  mCurrentFrame.MTimeFeatExtract;
   
    // For each extracted line get its parallel and perpendicular line correspondences    
    for (size_t k = 0; k < mCurrentFrame.mvLineEq.size(); k++)
    {
        std::vector<int> v_idx_perp;
        std::vector<int> v_idx_par;

        mCurrentFrame.mvPerpLinesIdx[k] = new std::vector<int>(v_idx_perp);
        mCurrentFrame.mvParLinesIdx[k] = new std::vector<int>(v_idx_par);
        
        // Discard non-valid lines   
        if (mCurrentFrame.mvLineEq[k][2] == 0.0)
            continue;
      
        mpManh->computeStructConstrains(mCurrentFrame, k, v_idx_par, v_idx_perp);

        mCurrentFrame.mvPerpLinesIdx[k] = new std::vector<int>(v_idx_perp);
        mCurrentFrame.mvParLinesIdx[k] = new std::vector<int>(v_idx_par);
    }

    std::chrono::steady_clock::time_point t_st_line_opt = std::chrono::steady_clock::now();

    // Optimize line-endpoints using structural constraints
    Optimizer::LineOptStruct(&mCurrentFrame);

    std::chrono::steady_clock::time_point t_end_line_opt = std::chrono::steady_clock::now();
    chrono::duration<double> time_line_opt = chrono::duration_cast<chrono::duration<double>>(t_end_line_opt - t_st_line_opt);
    mSumMTimeEptsLineOpt += time_line_opt.count();

    Track();

    return mCurrentFrame.mTcw.clone();
}

bool Tracking::ExtractCoarseManhAx()
{
    // Compute the Coarse Manhattan Axis
    float succ_rate = -1.0;

    // TODO 0: Avoid this conversion to improve efficiency
    std::vector<cv::Mat> lines_vector;
    for (size_t i = 0; i < mCurrentFrame.mvLineEq.size(); i++)
    {
        if (mCurrentFrame.mvLineEq[i][2] == -1.0 ||
            mCurrentFrame.mvLineEq[i][2] == 1.0)
            continue;

        cv::Mat line_vector = (Mat_<double>(3, 1)
                                   << mCurrentFrame.mvLineEq[i][0],
                               mCurrentFrame.mvLineEq[i][1],
                               mCurrentFrame.mvLineEq[i][2]);

        lines_vector.push_back(line_vector);
    }

    // FindCoordAxis function produces a set of initial candidate directions by evaluating line vectors and point normals.
    // This directions are used to estimate the course Manh. axes.

    // TODO 0: mCurrentFrame.mRepLines and mCurrentFrame.mRepNormals contains a vector of vector. Mod.
    std::vector<cv::Mat> coord_cand_lines;
    mpManh->findCoordAxis(mCurrentFrame.mRepLines, coord_cand_lines);

    std::vector<cv::Mat> coord_cand;
    mpManh->findCoordAxis(mCurrentFrame.mRepNormals, coord_cand);

    std::vector<cv::Mat> v_lines_n_normals(mCurrentFrame.mvPtNormals);
    v_lines_n_normals.insert(v_lines_n_normals.end(), lines_vector.begin(), lines_vector.end());

    std::vector<cv::Mat> v_lines_n_normals_cand(coord_cand);
    v_lines_n_normals_cand.insert(v_lines_n_normals_cand.end(), coord_cand_lines.begin(), coord_cand_lines.end());
    
    cv::Mat manh_axis;
    if (mpManh->extractCoarseManhAxes(v_lines_n_normals, v_lines_n_normals_cand, manh_axis, succ_rate) && succ_rate > 0.95)
    {
        // Assign the extracted Manh. axes to frame lines
        std::vector<int> line_axis_corresp;
        mpManh->LineManhAxisCorresp(manh_axis, mCurrentFrame.mvLineEq, line_axis_corresp);
        mCurrentFrame.vManhAxisIdx = line_axis_corresp;
        std::cerr << "EXTRACTED MANH AXES with succes rate: " << succ_rate  << std::endl;
        mpMap->SetWorldManhAxis(manh_axis.clone());
        mpLocalMapper->setManhAxis(manh_axis.clone());
        return true;
    } 
    return false;
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    static int count=0;
    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
    {
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpLSDextractorLeft,mpORBVocabulary,mpManh,mK,mDistCoef,mbf,mThDepth,mask);
    }
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpLSDextractorLeft,mpORBVocabulary,mpManh, mK,mDistCoef,mbf,mThDepth,mask);
    
    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD){

            std::chrono::steady_clock::time_point t_st_coarse_manh = std::chrono::steady_clock::now();

            if (ExtractCoarseManhAx())
                mCoarseManhInit = true;
            else
            {
                std::cerr << "WARNING -- Not able to seek manh init" << std::endl;
                // Assign lines to Manh. Axes 0, which means that do not correspond to a Manh. Axis
                std::vector<int> v_zeros(mCurrentFrame.mvLineEq.size(), 0);
                mCurrentFrame.vManhAxisIdx = v_zeros;
            }

            std::chrono::steady_clock::time_point t_end_coarse_manh = std::chrono::steady_clock::now();
            chrono::duration<double> time_coarse_manh = chrono::duration_cast<chrono::duration<double>>(t_end_coarse_manh - t_st_coarse_manh);
            mTimeCoarseManh += time_coarse_manh.count();
            StereoInitialization();
        }
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // If coarse Manh. Axes has not been extracted, try to extract it
        if(!mCoarseManhInit)
            if (ExtractCoarseManhAx())
                mCoarseManhInit = true;

        // Evaluate if the Local Mapper has computed the fine Manhattan Axes
        if (!mManhInit && mCoarseManhInit)
        {
            cv::Mat opt_manh_wm;

            if (mpLocalMapper->optManhInitAvailable(opt_manh_wm))
            {
                mManhInit = true; 
                // Update the coarse Manh. axes from the map     
                mpMap->SetWorldManhAxis(opt_manh_wm.clone());
                
            }
        }

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        std::chrono::steady_clock::time_point t_st_cam_pose_estim = std::chrono::steady_clock::now();
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if(mState == 3)
            {
                std::cerr << " << Error: State 3 Tracking Lost. Press enter to finish the process" << std::endl;
                std::cin.get();
            }
        
            if(mState==OK)
            {   
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();
           
                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

                    bOK = TrackWithMotionModel();

                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated
            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    
                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }
        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                bOK = TrackLocalMapWithLines();
            }

        }
        else
        {
            if(bOK && !mbVO)
                bOK = TrackLocalMapWithLines();
        }

        std::chrono::steady_clock::time_point t_end_cam_pose_estim  = std::chrono::steady_clock::now();
        chrono::duration<double> t_time_cam_pose_est = chrono::duration_cast<chrono::duration<double>>(t_end_cam_pose_estim -t_st_cam_pose_estim);
        mSumTimePoseEstim += t_time_cam_pose_est.count();

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking was good, check if it should be inserted as a keyframe
        if(bOK)
        {
            // Assign the Manh. axes to each line. If not assign 0, which means non-associated axis
            if (!(mpMap->GetWorldManhAxis().empty()))
            {
                cv::Mat frame_mRcw;
                mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3).copyTo(frame_mRcw);
                frame_mRcw.convertTo(frame_mRcw, CV_64F);
                std::vector<int> line_axis_corresp(mCurrentFrame.mvLineEq.size(), 0);
                mpManh->LineManhAxisCorresp(frame_mRcw, mCurrentFrame.mvLineEq, line_axis_corresp);
                mCurrentFrame.vManhAxisIdx = line_axis_corresp;
            }
            else
            {
                std::vector<int> v_zeros(mCurrentFrame.mvLineEq.size(), 0);
                mCurrentFrame.vManhAxisIdx = v_zeros;
            }

            // Update motion model
            if (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
            }

            for (int i = 0; i < mCurrentFrame.NL; i++)
            {
                MapLine *pML = mCurrentFrame.mvpMapLines[i];
                if (pML)
                    if (pML->Observations() < 1)
                    {
                        mCurrentFrame.mvbLineOutlier[i] = false;
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();
            // Check if we need to insert a new keyframe
            if (NeedNewKeyFrame())
            {
                CreateNewKeyFrame();
            }

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }

            for(int i=0; i<mCurrentFrame.NL; i++)
            {
                if(mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                    mCurrentFrame.mvpMapLines[i]= static_cast<MapLine*>(NULL);
            }
        }
        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }
        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
}

void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N + mCurrentFrame.NL > 100)
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);
        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }
        for(int i=0; i<mCurrentFrame.NL;i++)
        {
            if(mCurrentFrame.mvKeylinesUn[i].startPointX > 0.0 && mCurrentFrame.mvLines3D[i].first.x() !=0)
            {
                Eigen::Vector3d st_3D_w = mCurrentFrame.PtToWorldCoord(mCurrentFrame.mvLines3D[i].first);
                Eigen::Vector3d e_3D_w = mCurrentFrame.PtToWorldCoord(mCurrentFrame.mvLines3D[i].second);
                Vector6d w_l_endpts;
                w_l_endpts << st_3D_w.x(), st_3D_w.y(),st_3D_w.z(),
                e_3D_w.x(), e_3D_w.y(),e_3D_w.z();

                MapLine* pML = new MapLine(w_l_endpts,mCurrentFrame.vManhAxisIdx[i], pKFini, mpMap);
                pML->AddObservation(pKFini, i);
                pKFini->AddMapLine(pML, i);
                pML->ComputeDistinctiveDescriptors();
                pML->UpdateManhAxis();
                mpMap->AddMapLine(pML);
                mCurrentFrame.mvpMapLines[i]=pML;
                }
        }

        cout << "KF0 - New map created with " << mpMap->MapPointsInMap() << " points" << endl;
        cout << "KF0 - New map created with " << mpMap->MapLinesInMap() << " lines" << endl << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mvpLocalMapLines=mpMap->GetAllMapLines();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{
    int num = 100;
    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>num)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            fill(mvIniLineMatches.begin(),mvIniLineMatches.end(),-1);
            mvIniLastLineMatches = vector<int>(mCurrentFrame.mvKeys.size(), -1);

            mbIniFirst = false;

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=num)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            fill(mvIniLineMatches.begin(),mvIniLineMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        LSDmatcher lmatcher;   
        int lineMatches = lmatcher.SearchDouble(mLastFrame, mCurrentFrame, mvIniLineMatches);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        if(!mbIniFirst)
        {
            mvIniLastLineMatches = mvIniLineMatches;
            mbIniFirst = true;
        }else{
            for(int i = 0; i < mInitialFrame.mvKeys.size(); i++)
            {
                int j = mvIniLastLineMatches[i];
                if(j >= 0 ){
                    mvIniLastLineMatches[i] = mvIniLineMatches[j];
                }
            }

            lmatcher.SearchDouble(mInitialFrame,mCurrentFrame, mvIniLineMatches);
            for(int i = 0; i < mInitialFrame.mvKeys.size(); i++)
            {
                int j = mvIniLastLineMatches[i];
                int k = mvIniLineMatches[i];
                if(j != k){
                    mvIniLastLineMatches[i] = -1;
                }
            }
        }

        mvIniLineMatches = mvIniLastLineMatches;

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
        vector<bool> mvbLineTriangulated; 

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated, mvIniLineMatches, mvLineS3D, mvLineE3D, mvbLineTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            for(size_t i=0, iend=mvIniLineMatches.size(); i<iend;i++)
            {
                if(mvIniLineMatches[i]>=0 && !mvbLineTriangulated[i])
                {
                    mvIniLineMatches[i]=-1;
                    lineMatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);
           
            CreateInitialMapMonoWithLine();
        }

        mLastFrame = Frame(mCurrentFrame);
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        mpMap->AddMapPoint(pMP);
    }

    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);
    mState=OK;  
}

void Tracking::CreateInitialMapMonoWithLine()
{
    KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    for(size_t i=0; i<mvIniMatches.size(); i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        // Create MapPoint
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);
        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        // Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        // Add to Map
        mpMap->AddMapPoint(pMP);
    }

    for(size_t i=0; i<mvIniLineMatches.size(); i++)
    {
        if(mvIniLineMatches[i] < 0)
            continue;

        // Create MapLine
        Vector6d worldPos;
        worldPos << mvLineS3D[i].x, mvLineS3D[i].y, mvLineS3D[i].z, mvLineE3D[i].x, mvLineE3D[i].y, mvLineE3D[i].z;

        int manh_idx = 0;
        MapLine* pML = new MapLine(worldPos, manh_idx, pKFcur, mpMap);

        pKFini->AddMapLine(pML,i);
        pKFcur->AddMapLine(pML,mvIniLineMatches[i]);

        pML->AddObservation(pKFini, i);
        pML->AddObservation(pKFcur, mvIniLineMatches[i]);

        pML->ComputeDistinctiveDescriptors();

        pML->UpdateAverageDir();

        // Fill Current Frame structure
        mCurrentFrame.mvpMapLines[mvIniLineMatches[i]] = pML;
        mCurrentFrame.mvbLineOutlier[mvIniLineMatches[i]] = false;

        // step5.4: Add to Map
        mpMap->AddMapLine(pML);
    }

    cout << "this Map created with " << mpMap->MapPointsInMap() << " points, and "<< mpMap->MapLinesInMap() << " lines." << endl;
    Optimizer::GlobalBundleAdjustemnt(mpMap, 20, true); 
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    cout << "medianDepth = " << medianDepth << endl;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<80)
    {
        cout << "Wrong initialization, reseting ... " << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale Points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); ++iMP)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    // Scale Line Segments
    vector<MapLine*> vpAllMapLines = pKFini->GetMapLineMatches();
    for(size_t iML=0; iML < vpAllMapLines.size(); iML++)
    {
        if(vpAllMapLines[iML])
        {
            MapLine* pML = vpAllMapLines[iML];
            pML->SetWorldPos(pML->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mvpLocalMapLines = mpMap->GetAllMapLines();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    mpMap->SetReferenceMapLines(mvpLocalMapLines);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
    for(int i=0; i<mLastFrame.NL; i++)
    {
        MapLine* pML = mLastFrame.mvpMapLines[i];

        if(pML)
        {
            MapLine* pReL = pML->GetReplaced();
            if(pReL)
            {
                mLastFrame.mvpMapLines[i] = pReL;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{   
    cout<<"[Debug] Calling TrackReferenceKeyFrameWithLine(), mCurrentFrame.mnId:"<<mCurrentFrame.mnId<<endl;

    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;
    
    fill(mCurrentFrame.mvpMapLines.begin(),mCurrentFrame.mvpMapLines.end(),static_cast<MapLine*>(NULL));

    LSDmatcher lmatcher(0.85, true);

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);
    int lmatches = 0;

    std::vector<int> matches_12;
    int nl_matches = lmatcher.match(mLastFrame.mLdesc, mCurrentFrame.mLdesc, 0.9, matches_12);

    const double deltaAngle = M_PI/8.0;
    const double deltaWidth = (mCurrentFrame.mnMaxX-mCurrentFrame.mnMinX)*0.1;
    const double deltaHeight = (mCurrentFrame.mnMaxY-mCurrentFrame.mnMinY)*0.1;
    int delta_angle = 0;
    int delta_pose = 0;
    int not_found = 0;
    int i2_var = 0;
    const int nmatches_12 = matches_12.size();
    for (int i1 = 0; i1 < nmatches_12; ++i1) {
        if(!mLastFrame.mvpMapLines[i1]){
          not_found ++;
          continue;  
        } 
        const int i2 = matches_12[i1];
        if (i2 < 0) 
        {
            i2_var ++;
            continue;
        }
        
        if(mCurrentFrame.mvKeylinesUn[i2].startPointX == 0) continue; 

        // Evaluate orientation and position in image
        if(true) {
            // Orientation
            double theta = mCurrentFrame.mvKeylinesUn[i2].angle-mLastFrame.mvKeylinesUn[i1].angle;
            if(theta<-M_PI) theta+=2*M_PI;
            else if(theta>M_PI) theta-=2*M_PI;
            if(fabs(theta)>deltaAngle) {
                matches_12[i1] = -1;
                delta_angle++;
                continue;
            }
            
            // Position
            const float& sX_curr = mCurrentFrame.mvKeylinesUn[i2].startPointX;
            const float& sX_last = mLastFrame.mvKeylinesUn[i1].startPointX;
            const float& sY_curr = mCurrentFrame.mvKeylinesUn[i2].startPointY;
            const float& sY_last = mLastFrame.mvKeylinesUn[i1].startPointY;
            const float& eX_curr = mCurrentFrame.mvKeylinesUn[i2].endPointX;
            const float& eX_last = mLastFrame.mvKeylinesUn[i1].endPointX;
            const float& eY_curr = mCurrentFrame.mvKeylinesUn[i2].endPointY;
            const float& eY_last = mLastFrame.mvKeylinesUn[i1].endPointY;
            if(fabs(sX_curr-sX_last)>deltaWidth || fabs(eX_curr-eX_last)>deltaWidth || fabs(sY_curr-sY_last)>deltaHeight || fabs(eY_curr-eY_last)>deltaHeight )
            {
                matches_12[i1] = -1;
                delta_pose ++;
                continue;
            }
        }

        mCurrentFrame.mvpMapLines[i2] = mLastFrame.mvpMapLines[i1];
        ++lmatches;
    }

    if(nmatches<15  && lmatches<5)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;

    // Pose optimization using the reprojection error of 3D-2D
    mCurrentFrame.SetPose(mLastFrame.mTcw);
    
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard Point outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    // Discard Line outliers
    int nLinematchesMap = 0;
    for(int i =0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(mCurrentFrame.mvbLineOutlier[i])
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];

                mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                mCurrentFrame.mvbLineOutlier[i]=false;
                pML->mbTrackInView = false;
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                lmatches--;
            }
            else if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                nLinematchesMap++;
        }

        mCurrentFrame.mvbLineOutlier[i] = false;
    }
    int del_par_track = 0;

    return nmatchesMap>=10;
}


void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }

    // TODO 1: When the SLAM procedure will be ready in the future, add lines for the relocalisation mode .
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);
    LSDmatcher lmatcher;

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
   
    // Match Lines: Two options
    int lmatches = 0;
    
    float radius_th = 3.0;

    // 1/ Search by projection combining geometrical and appearance constraints
    // lmatches  = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame, radius_th);

    // 2/ Option from Ruben Gomez Ojeda -- line segments f2f tracking
    float des_th = 0.95;
    lmatches  = lmatcher.SearchByGeomNApearance(mCurrentFrame, mLastFrame, des_th);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    // fill(mCurrentFrame.mvpMapLines.begin(),mCurrentFrame.mvpMapLines.end(),static_cast<MapLine*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;

    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if((nmatches + lmatches) <20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
        lmatches  = lmatcher.SearchByProjection(mCurrentFrame, mLastFrame, 2*radius_th);
    }

    if(nmatches<20 && lmatches<5)
        return false;

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard Pt outliers
    int n_matches_map = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                n_matches_map++;
        }
    }    

    // Discard Line outliers
    int nLinematchesMap = 0;
    for(int i =0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(mCurrentFrame.mvbLineOutlier[i])
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];

                mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                mCurrentFrame.mvbLineOutlier[i]=false;
                pML->mbTrackInView = false;
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                lmatches--;
            }
            else if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                nLinematchesMap++;
        }
        mCurrentFrame.mvbLineOutlier[i] = false;
    }

    int n_matches_map_pts_lines =  n_matches_map + nLinematchesMap;

    int nmatches_pts_lines = nmatches + lmatches;

    if(mbOnlyTracking)
    {
        mbVO = n_matches_map_pts_lines<20;
        return nmatches_pts_lines>20;
    }

    return n_matches_map_pts_lines>=20;
}

bool Tracking::TrackLocalMapWithLines()
{
    UpdateLocalMap();

    thread threadPoints(&Tracking::SearchLocalPoints, this);
    thread threadLines(&Tracking::SearchLocalLines, this);
    threadPoints.join();
    threadLines.join();
       
    std::chrono::steady_clock::time_point t1_line_opt = std::chrono::steady_clock::now();

    mpManh->computeStructConstInMap(mCurrentFrame, mvpLocalMapLines_InFrustum);
      std::chrono::steady_clock::time_point t2_line_opt = std::chrono::steady_clock::now();
    chrono::duration<double> time_used_line_opt = chrono::duration_cast<chrono::duration<double>>(t2_line_opt - t1_line_opt);

    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;
    mnLineMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Update MapLines Statistics
    for(int i=0; i<mCurrentFrame.NL; i++)
    {
        if(mCurrentFrame.mvpMapLines[i])
        {
            if(!mCurrentFrame.mvbLineOutlier[i])
            {
                mCurrentFrame.mvpMapLines[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapLines[i]->Observations()>0)
                        mnLineMatchesInliers++;
                }
                else
                    mnLineMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapLines[i] = static_cast<MapLine*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers + mnLineMatchesInliers <50)
        return false;

    if(mnMatchesInliers + mnLineMatchesInliers <15)
        return false;
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints and MapLines in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);
    int nRefMatchesLines = mpReferenceKF->TrackedMapLines(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points and lines are being tracked and how many could be potentially created.
    // This stage differs from ORB-SLAM2, we use the ratio, because it better adapts to low-textured environments. 
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    float ratio_pts = nNonTrackedClose / float(nTrackedClose + nNonTrackedClose);

    int nNonTrackedCloseLine = 0;
    int nTrackedCloseLine= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.NL; i++)
        {
            if(mCurrentFrame.mvLines3D[i].first[2]>0 && mCurrentFrame.mvLines3D[i].first[2]<mThDepth)
            {
                if(mCurrentFrame.mvpMapLines[i] && !mCurrentFrame.mvbLineOutlier[i])
                    nTrackedCloseLine++;
                else
                    nNonTrackedCloseLine++;
            }
        }
    }

    float ratio_lines = nNonTrackedCloseLine / float(nTrackedCloseLine + nNonTrackedCloseLine);
    bool bNeedToInsertClosePtsLine = ratio_pts >0.6 || ratio_lines > 0.6;
    
    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=(mnLastKeyFrameId + 30)) && bLocalMappingIdle);
   
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.4 || bNeedToInsertClosePtsLine || mnLineMatchesInliers < nRefMatchesLines*0.4) ;

    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClosePtsLine) && mnMatchesInliers>15) || ((mnLineMatchesInliers<nRefMatchesLines*thRefRatio|| bNeedToInsertClosePtsLine) && mnLineMatchesInliers>15));

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            
                nPoints++;

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }
     if(mSensor!=System::MONOCULAR)
    {
        vector<pair<float, int>> vDepthIdxLines;
        vDepthIdxLines.reserve(mCurrentFrame.NL);
        for (int i = 0; i < mCurrentFrame.NL; i++)
        {
            if(mCurrentFrame.mvLines3D[i].first.z() == 0 || mCurrentFrame.mvLines3D[i].second.z()  == 0)
                continue;

            double sz = mCurrentFrame.mvLines3D[i].first.z();
            double ez = mCurrentFrame.mvLines3D[i].second.z();
            float z = sz > ez ? sz : ez;
            vDepthIdxLines.push_back(make_pair(z, i));
        }
       
        if (!vDepthIdxLines.empty())
        {
            sort(vDepthIdxLines.begin(), vDepthIdxLines.end());

            int nLines = 0;
            for (size_t j = 0; j < vDepthIdxLines.size(); j++)
            {
                int i = vDepthIdxLines[j].second;

                bool bCreateNew = false;

                MapLine *pML = mCurrentFrame.mvpMapLines[i];
                if (!pML)
                    bCreateNew = true;
                else if (pML->Observations() < 1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                }

                if (bCreateNew)
                {
                    // Select a valid Line
                    if (mCurrentFrame.mvLines3D[i].first.z() != 0.0)
                    {
                        Eigen::Vector3d st_3D_w = mCurrentFrame.PtToWorldCoord(mCurrentFrame.mvLines3D[i].first);
                        Eigen::Vector3d e_3D_w = mCurrentFrame.PtToWorldCoord(mCurrentFrame.mvLines3D[i].second);

                        Vector6d w_l_endpts;
                        w_l_endpts << st_3D_w.x(), st_3D_w.y(), st_3D_w.z(),
                            e_3D_w.x(), e_3D_w.y(), e_3D_w.z();

                       MapLine *pNewML = new MapLine(w_l_endpts, mCurrentFrame.vManhAxisIdx[i], pKF, mpMap);
                       pNewML->UpdateManhAxis();

                        pNewML->AddObservation(pKF, i);

                        std::vector<int> v_idx_perp;
                        std::vector<int> v_idx_par;

                        mpManh->computeStructConstrains(mCurrentFrame, pNewML, v_idx_par, v_idx_perp);

                        if (mCurrentFrame.mvParallelLines[i]->size() > 0)
                        {
                            pNewML->AddParObservation(pKF, v_idx_par);
                        }

                        if (mCurrentFrame.mvPerpLines[i]->size() > 0)
                        {
                            pNewML->AddPerpObservation(pKF, v_idx_perp);
                        }
                        pKF->AddMapLine(pNewML, i);
                        pNewML->ComputeDistinctiveDescriptors();
                        // TODO 0: check if the two following lines are required 
                        // pNewML-> UpdateAverageDir();
                        //  pNewML->UpdateNormalAndDepth();
                        mpMap->AddMapLine(pNewML);

                        mCurrentFrame.mvpMapLines[i] = pNewML;
                        nLines++;
                    }
                    else
                    {
                        nLines++;
                    }

                    if (vDepthIdxLines[j].first > mThDepth && nLines > 100)
                        break;
                }
             }
        }


    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::SearchLocalLines()
{
    bool eval_orient = true;
    if(mSensor==System::MONOCULAR)
    {
        eval_orient = false;
    }

    // vector<MapLine*> mvpLocalMapLines_InFrustum;
    for(vector<MapLine*>::iterator vit=mCurrentFrame.mvpMapLines.begin(), vend=mCurrentFrame.mvpMapLines.end(); vit!=vend; vit++)
    {
        MapLine* pML = *vit;
        if(pML)
        {
            if(pML->isBad())
            {
                *vit = static_cast<MapLine*>(NULL);
            } 
            else{
                pML->IncreaseVisible();
                pML->mnLastFrameSeen = mCurrentFrame.mnId;
                pML->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;
    mvpLocalMapLines_InFrustum.clear(); 

    for (vector<MapLine *>::iterator vit = mvpLocalMapLines.begin(), vend = mvpLocalMapLines.end(); vit != vend; vit++)
    {
        MapLine *pML = *vit;
        if (pML->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pML->isBad())
            continue;

         // Project (this fills MapLine variables for matching)
        if (mCurrentFrame.isInFrustum(pML, 0.5))
        {
            pML->IncreaseVisible();
            nToMatch++;
            mvpLocalMapLines_InFrustum.push_back(pML);
        }
    }

    if(nToMatch>0)
    {
        LSDmatcher matcher;
        int th = 1;

        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        int nmatches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, eval_orient, th);

        if(nmatches)
        {
            for(int i = 0; i<mCurrentFrame.mvpMapLines.size(); i++)
            {
                MapLine* pML = mCurrentFrame.mvpMapLines[i];
                if(pML)
                {
                    Eigen::Vector3d tWorldVector = pML->GetWorldVector();
                    cv::Mat tWorldVector_ = (cv::Mat_<float>(3, 1) << tWorldVector(0), tWorldVector(1), tWorldVector(2));
                    KeyLine tkl = mCurrentFrame.mvKeylinesUn[i];
                    cv::Mat tklS = (cv::Mat_<float>(3, 1) << tkl.startPointX, tkl.startPointY, 1.0);
                    cv::Mat tklE = (cv::Mat_<float>(3, 1) << tkl.endPointX, tkl.endPointY, 1.0);
                    cv::Mat K = mCurrentFrame.mK;
                    cv::Mat tklS_ = K.inv() * tklS; cv::Mat tklE_ = K.inv() * tklE;

                    cv::Mat NormalVector_ = tklS_.cross(tklE_);
                    double norm_ = cv::norm(NormalVector_);
                    NormalVector_ /= norm_;

                    cv::Mat Rcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
                    cv::Mat tCameraVector_ = Rcw * tWorldVector_;
                    double CosSita = abs(NormalVector_.dot(tCameraVector_));

                    if(CosSita>0.09)
                    {
                        mCurrentFrame.mvpMapLines[i]=static_cast<MapLine*>(NULL);
                    }
                }
            }
        }

    }

}

void Tracking::UpdateLocalMap()
{
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
    
    // TODO: 0 Join Lines, perform it in the back-end
    // std::vector<std::vector<int>> sim_lines_idx;
    // FindSimilarLines(sim_lines_idx);
    // JoinLines(sim_lines_idx);

    mpMap->SetReferenceMapLines(mvpLocalMapLines);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
    UpdateLocalLines();
}

void Tracking::FindSimilarLines(std::vector<std::vector<int>> &sim_lines_idx)
{
    double angle = 10;
    double th_rad = angle / 180.0 * M_PI;
    double th_angle = std::cos(th_rad);

    std::vector<bool> found_idx;
    found_idx.resize(mvpLocalMapLines.size(), false);

    for (size_t i = 0; i < mvpLocalMapLines.size(); i++)
    {
        if(found_idx[i])
            continue;

        Vector6d v1 = mvpLocalMapLines[i]->GetWorldPos();
        cv::Mat sp1 = (Mat_<double>(3, 1) << v1(0), v1(1), v1(2));
        cv::Mat ep1 = (Mat_<double>(3, 1) << v1(3), v1(4), v1(5));
        cv::Mat l1 = ep1 - sp1;

        std::vector<int> sim_l_idx;

        sim_l_idx.push_back(i);

        for (size_t j = i + 1; j < mvpLocalMapLines.size(); j++)
        {
            if(found_idx[j])
            continue;

            // Descriptor distance
            int desc_dist = norm(mvpLocalMapLines[i]->GetDescriptor(), mvpLocalMapLines[j]->GetDescriptor(), NORM_HAMMING);

            if (desc_dist > 100)
                continue;

            Vector6d v2 = mvpLocalMapLines[j]->GetWorldPos();

            cv::Mat sp2 = (Mat_<double>(3, 1) << v2(0), v2(1), v2(2));
            cv::Mat ep2 = (Mat_<double>(3, 1) << v2(3), v2(4), v2(5));
            cv::Mat l2 = ep2 - sp2;
          
            // Evaluate orientation
            double angle = mpLSDextractorLeft->computeAngle(l1, l2);

            if (angle < th_angle)
                continue;

            float pt_l_dist = PointToLineDist(sp1, ep1, sp2, ep2);

            if (pt_l_dist > 0.02)
                continue;

            // Evaluate euclidean distance between pts
            cv::Mat m_distances = (Mat_<float>(1, 4) << cv::norm(sp1 - sp2), cv::norm(ep1 - ep2), cv::norm(sp1 - ep2), cv::norm(ep1 - sp2));

            // Get minimum value;
            double min, max;
            cv::minMaxLoc(m_distances, &min, &max);

            if (min > 0.2)
                continue;

            sim_l_idx.push_back(j);
            found_idx[j] = true;
        }

        if(sim_l_idx.size()>1)
            sim_lines_idx.push_back(sim_l_idx);
    }
}

void Tracking::JoinLines(const std::vector<std::vector<int>> &sim_lines_idx)
{
    for (size_t i = 0; i < sim_lines_idx.size(); i++)
    {
        std::vector<int> single_sim_lines = sim_lines_idx[i];
      
        int rep_idx = ComputeRepresentMapLine(single_sim_lines);
        int idx_desc = FindRepresentDesc(single_sim_lines);

        // Evaluates if a representative IDX is found
        if(rep_idx < 0)
            continue;

        for (size_t j = 0; j < single_sim_lines.size(); j++)
        {
            if(single_sim_lines[j] == rep_idx)
            {

                if (rep_idx != idx_desc)
                mvpLocalMapLines[rep_idx]->mLDescriptor = mvpLocalMapLines[idx_desc]->GetDescriptor();
                continue;
            }
            mvpLocalMapLines[single_sim_lines[j]]->SetBadFlag();
        }
    }

}

int Tracking::FindRepresentDesc(std::vector<int> v_idxs)
{
    int nl = v_idxs.size();
        std::vector<std::vector<float>> Distances;
        Distances.resize(nl, vector<float>(nl, 0));
        for(size_t i=0; i<nl; i++)
        {
            Distances[i][i]=0;
            for(size_t j=0; j<nl; j++)
            {
                int distij = norm(mvpLocalMapLines[v_idxs[i]]->GetDescriptor(), mvpLocalMapLines[v_idxs[j]]->GetDescriptor(), NORM_HAMMING);

                Distances[i][j]=distij;
                Distances[j][i]=distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for(size_t i=0; i<nl; i++)
        {
            vector<int> vDists(Distances[i].begin(), Distances[i].end());
            sort(vDists.begin(), vDists.end());

            int median = vDists[0.5*(nl-1)];

            if(median<BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }
        return v_idxs[BestIdx];
        // {
        //     unique_lock<mutex> lock(mMutexFeatures);
        //     mLDescriptor = vDescriptors[BestIdx].clone();
        // }
}

int Tracking::ComputeRepresentMapLine(const std::vector<int> &v_idxs)
{
    std::vector<double> lengths;
    lengths.resize(v_idxs.size(), 0);

    double max_length = 0.0;
    int max_length_idx = -1;

    for (size_t i = 0; i < v_idxs.size(); i++)
    {
        Vector6d pts = mvpLocalMapLines[v_idxs[i]]->GetWorldPos();

        cv::Mat spt = (Mat_<float>(3, 1) << pts(0), pts(1), pts(2));
        cv::Mat ept = (Mat_<float>(3, 1) << pts(3), pts(4), pts(5));
        // compute length
        double line_length = abs(cv::norm(ept - spt));

        if(line_length> max_length)
        {
            max_length = line_length;
            max_length_idx = i;
        }

        lengths.push_back(line_length);
    }

   return(v_idxs[max_length_idx]);
}

float Tracking::PointToLineDist(const cv::Mat &sp1, const cv::Mat &ep1, const cv::Mat &sp2, const cv::Mat &ep2)
{

    cv::Mat v = sp1;
    cv::Mat ab = ep2 - sp2;
    cv::Mat av = v - sp2;
    if( av.dot(ab)<= 0.0f )
    {
        return(cv::norm(av));
    }
    cv::Mat bv = v -ep2;

    if( bv.dot(ab)<= 0.0f )
    {
        return(cv::norm(bv));
    }

    return (cv::norm(ab.cross( av ))) / (cv::norm(ab)) ;   
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalLines()
{
    mvpLocalMapLines.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapLine*> vpMLs = pKF->GetMapLineMatches();

        for(vector<MapLine*>::const_iterator itML=vpMLs.begin(), itEndML=vpMLs.end(); itML!=itEndML; itML++)
        {
            MapLine* pML = *itML;
            if(!pML)
                continue;
            if(pML->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pML->isBad())
            {
                mvpLocalMapLines.push_back(pML);
                pML->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    std::cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    std::cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    std::cout << " done" << endl;

    // Reset Loop Closing
    std::cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    std::cout << " done" << endl;

    // Clear BoW Database
    std::cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    std::cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} //namespace ORB_SLAM
