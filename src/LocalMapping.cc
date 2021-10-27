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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

#define PI 3.1415926

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true), mAvailableOptManh(false)
{
    mTimeLocalMapOpt = 0.0;
    mTimeFineManhInit = 0.0;
    mNumKFsComp = 0;
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;
    int obt_manh_kf_id = -1;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false);

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            ProcessNewKeyFrame();
            // Save the KF id where the coarse Manh. Axes is extracted
            if(obt_manh_kf_id == -1 && !(mManhAxis.empty()))
            {
                obt_manh_kf_id = mpMap->KeyFramesInMap();
            }

            thread threadCullPoint(&LocalMapping::MapPointCulling, this);
            thread threadCullLine(&LocalMapping::MapLineCulling, this);
            threadCullPoint.join();
            threadCullLine.join();

            thread threadCreateP(&LocalMapping::CreateNewMapPoints, this);
            // Option 1;
            // thread threadCreateL(&LocalMapping::CreateNewMapLines2, this);
            // Option 2:
            thread threadCreateL(&LocalMapping::CreateNewMapLinesConstraint, this);
            threadCreateP.join();
            threadCreateL.join();

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }
            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Conpute fine Manh. Axes after 4 keyframes from the coarse Manh. Axes extraction
                if (mpMap->KeyFramesInMap() - obt_manh_kf_id > 4 && mAvailableOptManh == false && obt_manh_kf_id != -1)
                {
                    std::chrono::steady_clock::time_point time_fine_manh_init_1 = std::chrono::steady_clock::now();
                    Optimizer::MultiViewManhInit(mManhAxis, mpCurrentKeyFrame, &mbAbortBA, mpMap); 
                    mAvailableOptManh = true;
                    std::chrono::steady_clock::time_point time_fine_manh_init_2 = std::chrono::steady_clock::now();

                    chrono::duration<double> time_fine_manh_init = chrono::duration_cast<chrono::duration<double>>(time_fine_manh_init_2 - time_fine_manh_init_1);
                    mTimeFineManhInit = time_fine_manh_init.count();
                }

                // VI-D Local BA
                if(mpMap->KeyFramesInMap()>3)
                {
                    // Contributions of this work
                    std::chrono::steady_clock::time_point time_local_map_opt1 = std::chrono::steady_clock::now();

                    Optimizer::LocalMapOptimization(mManhAxis, mpCurrentKeyFrame, &mbAbortBA, mpMap, mAvailableOptManh);     //包含线特征的局部BA

                    std::chrono::steady_clock::time_point time_local_map_opt2 = std::chrono::steady_clock::now();
                    chrono::duration<double> time_local_map_opt = chrono::duration_cast<chrono::duration<double>>(time_local_map_opt2 - time_local_map_opt1);
                    mTimeLocalMapOpt += time_local_map_opt.count();
                    mNumKFsComp ++;
                }
                KeyFrameCulling();
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs);
    mlNewKeyFrames.push_back(pKF);
    mbAbortBA=true;
}

bool LocalMapping::optManhInitAvailable(cv::Mat &manh_optim)
{
    if(mAvailableOptManh)
    {
        manh_optim = mManhAxis.clone();
        return true;
    }
    return false;
}

void LocalMapping::setManhAxis(cv::Mat manh_axis)
{
    mManhAxis = manh_axis;
}

bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs);
    return(!mlNewKeyFrames.empty());
}

void LocalMapping::ProcessNewKeyFrame()
{
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        mpCurrentKeyFrame = mlNewKeyFrames.front();
        mlNewKeyFrames.pop_front();
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for(size_t i=0; i<vpMapPointMatches.size(); i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i);
                    pMP->UpdateNormalAndDepth();
                    pMP->ComputeDistinctiveDescriptors();
                }
                else // Only used for new stereo points inserted by the Tracking
                {
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }

    const vector<MapLine*> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();

    for(size_t i=0; i<vpMapLineMatches.size(); i++)
    {
        MapLine* pML = vpMapLineMatches[i];
        if(pML)
        {
            if(!pML->isBad())
            {
                if(!pML->IsInKeyFrame(mpCurrentKeyFrame))
                {
                    pML->AddObservation(mpCurrentKeyFrame, i);  
                    pML->UpdateAverageDir();   
                    pML->ComputeDistinctiveDescriptors();
                } else
                {
                    mlpRecentAddedMapLines.push_back(pML);
                }
            }
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}


void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end())
    {
        MapPoint* pMP = *lit;
        if(pMP->isBad())
        {
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(pMP->GetFoundRatio()<0.25f )
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::MapLineCulling()
{
    // Check Recent Added MapLines
    list<MapLine*>::iterator lit = mlpRecentAddedMapLines.begin();
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId;

    int nThObs;
    if(mbMonocular)
        nThObs = 2;
    else
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapLines.end())
    {
        MapLine* pML = *lit;
        if(pML->isBad())
        {
            lit = mlpRecentAddedMapLines.erase(lit);
        }
        else if(pML->GetFoundRatio()<0.25f)
        {
            pML->SetBadFlag();
            lit = mlpRecentAddedMapLines.erase(lit);
        }
        // TODO 10: Adjust the parameters of the following 2 else if
        else if(((int)nCurrentKFid-(int)pML->mnFirstKFid)>=7 && pML->Observations()<=3)
        {
            pML->SetBadFlag();
            lit = mlpRecentAddedMapLines.erase(lit);
        }
        else if(((int)nCurrentKFid-(int)pML->mnFirstKFid)>=6)
            lit = mlpRecentAddedMapLines.erase(lit);
        else
            lit++;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular)
        nn=20;

    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;    
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)   
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);  
            const float ratioBaselineDepth = baseline/medianDepthKF2;  

            if(ratioBaselineDepth<0.01) 
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;  
            const int &idx2 = vMatchedIndices[ikp].second;  

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)   
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)  ///7.8和上面的5.991联系？
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);  

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

 void LocalMapping::CreateNewMapLines2() {
        // Retrieve neighbor keyframes in covisibility graph
        int nn = 10;
        if (mbMonocular)
            nn = 20;

        const vector<KeyFrame *> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

        LSDmatcher lmatcher;

        cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
        cv::Mat Rwc1 = Rcw1.t();
        cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
        cv::Mat Tcw1(3, 4, CV_32F);
        Rcw1.copyTo(Tcw1.colRange(0, 3));
        tcw1.copyTo(Tcw1.col(3));

        cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

        const Mat &K1 = mpCurrentKeyFrame->mK;
        const float &fx1 = mpCurrentKeyFrame->fx;
        const float &fy1 = mpCurrentKeyFrame->fy;
        const float &cx1 = mpCurrentKeyFrame->cx;
        const float &cy1 = mpCurrentKeyFrame->cy;
        const float &invfx1 = mpCurrentKeyFrame->invfx;
        const float &invfy1 = mpCurrentKeyFrame->invfy;

        const float ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

        int nnew = 0;

        // Search matches with epipolar restriction and triangulate
        for (size_t i = 0; i < vpNeighKFs.size(); i++) {
            if (i > 0 && CheckNewKeyFrames())
                return;

            KeyFrame *pKF2 = vpNeighKFs[i];

            // Check first that baseline is not too short
            cv::Mat Ow2 = pKF2->GetCameraCenter();
            cv::Mat vBaseline = Ow2 - Ow1;
            const float baseline = cv::norm(vBaseline);

            if (!mbMonocular) {
                if (baseline < pKF2->mb)
                    continue;
            } else {
                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                const float ratioBaselineDepth = baseline / medianDepthKF2;
                if (ratioBaselineDepth < 0.01)
                    continue;
            }

            // Compute Fundamental Matrix
            // Search matches that fulfill epipolar constraint
            vector<pair<size_t, size_t>> vMatchedIndices;
            lmatcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices);

            cv::Mat Rcw2 = pKF2->GetRotation();
            cv::Mat Rwc2 = Rcw2.t();
            cv::Mat tcw2 = pKF2->GetTranslation();
            cv::Mat Tcw2(3, 4, CV_32F);
            Rcw2.copyTo(Tcw2.colRange(0, 3));
            tcw2.copyTo(Tcw2.col(3));

            const Mat &K2 = pKF2->mK;
            const float &fx2 = pKF2->fx;
            const float &fy2 = pKF2->fy;
            const float &cx2 = pKF2->cx;
            const float &cy2 = pKF2->cy;
            const float &invfx2 = pKF2->invfx;
            const float &invfy2 = pKF2->invfy;

            // Triangulate each matched line Segment
            const int nmatches = vMatchedIndices.size();
            for (int ikl = 0; ikl < nmatches; ikl++) {
                const int &idx1 = vMatchedIndices[ikl].first;
                const int &idx2 = vMatchedIndices[ikl].second;

                const KeyLine &kl1 = mpCurrentKeyFrame->mvKeyLines[idx1];
                bool bStereo1 = mpCurrentKeyFrame->mvLineEq[idx1][2] > 0;

                const KeyLine &kl2 = pKF2->mvKeyLines[idx2];
                bool bStereo2 = mpCurrentKeyFrame->mvLineEq[idx2][2] > 0;

                cv::Mat kl1sp, kl1ep, kl2sp, kl2ep;
                kl1sp = (cv::Mat_<float>(3, 1) << (kl1.startPointX - cx1) * invfx1, (kl1.startPointY - cy1) *
                                                                                    invfy1, 1.0);
                kl1ep = (cv::Mat_<float>(3, 1) << (kl1.endPointX - cx1) * invfx1, (kl1.endPointY - cy1) * invfy1, 1.0);
                kl2sp = (cv::Mat_<float>(3, 1) << (kl2.startPointX - cx2) * invfx2, (kl2.startPointY - cy2) *
                                                                                    invfy2, 1.0);
                kl2ep = (cv::Mat_<float>(3, 1) << (kl2.endPointX - cx2) * invfx2, (kl2.endPointY - cy2) * invfy2, 1.0);

                cv::Mat sp3D, ep3D;
                if (bStereo1) {
                    Vector6d line3D = mpCurrentKeyFrame->obtain3DLine(idx1);
                    sp3D = cv::Mat::eye(3, 1, CV_32F);
                    ep3D = cv::Mat::eye(3, 1, CV_32F);
                    sp3D.at<float>(0) = line3D(0);
                    sp3D.at<float>(1) = line3D(1);
                    sp3D.at<float>(2) = line3D(2);
                    ep3D.at<float>(0) = line3D(3);
                    ep3D.at<float>(1) = line3D(4);
                    ep3D.at<float>(2) = line3D(5);
                } else if (bStereo2) {
                    Vector6d line3D = pKF2->obtain3DLine(idx2);
                    sp3D = cv::Mat::eye(3, 1, CV_32F);
                    ep3D = cv::Mat::eye(3, 1, CV_32F);
                    sp3D.at<float>(0) = line3D(0);
                    sp3D.at<float>(1) = line3D(1);
                    sp3D.at<float>(2) = line3D(2);
                    ep3D.at<float>(0) = line3D(3);
                    ep3D.at<float>(1) = line3D(4);
                    ep3D.at<float>(2) = line3D(5);
                } else
                    continue; //No stereo and very low parallax

                cv::Mat sp3Dt = sp3D.t();
                cv::Mat ep3Dt = ep3D.t();


                //Check triangulation in front of cameras
                float zsp1 = Rcw1.row(2).dot(sp3Dt) + tcw1.at<float>(2);
                if (zsp1 <= 0)
                    continue;

                float zep1 = Rcw1.row(2).dot(ep3Dt) + tcw1.at<float>(2);
                if (zep1 <= 0)
                    continue;

                float zsp2 = Rcw2.row(2).dot(sp3Dt) + tcw2.at<float>(2);
                if (zsp2 <= 0)
                    continue;

                float zep2 = Rcw2.row(2).dot(ep3Dt) + tcw2.at<float>(2);
                if (zep2 <= 0)
                    continue;

                //Check reprojection error in first keyframe
                const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kl1.octave];
                const float xsp1 = Rcw1.row(0).dot(sp3Dt) + tcw1.at<float>(0);
                const float ysp1 = Rcw1.row(1).dot(sp3Dt) + tcw1.at<float>(1);
                const float invzsp1 = 1.0 / zsp1;

                float usp1 = fx1 * xsp1 * invzsp1 + cx1;
                float vsp1 = fy1 * ysp1 * invzsp1 + cy1;

                float errXsp1 = usp1 - kl1.startPointX;
                float errYsp1 = vsp1 - kl1.startPointY;
                if ((errXsp1 * errXsp1 + errYsp1 * errYsp1) > 5.991 * sigmaSquare1)
                    continue;

                const float xep1 = Rcw1.row(0).dot(ep3Dt) + tcw1.at<float>(0);
                const float yep1 = Rcw1.row(1).dot(ep3Dt) + tcw1.at<float>(1);
                const float invzep1 = 1.0 / zep1;

                float uep1 = fx1 * xep1 * invzep1 + cx1;
                float vep1 = fy1 * yep1 * invzep1 + cy1;

                float errXep1 = uep1 - kl1.endPointX;
                float errYep1 = vep1 - kl1.endPointY;
                if ((errXep1 * errXep1 + errYep1 * errYep1) > 5.991 * sigmaSquare1)
                    continue;

                //Check reprojection error in second keyframe
                const float sigmaSquare2 = pKF2->mvLevelSigma2[kl2.octave];
                const float xsp2 = Rcw2.row(0).dot(sp3Dt) + tcw2.at<float>(0);
                const float ysp2 = Rcw2.row(1).dot(sp3Dt) + tcw2.at<float>(1);
                const float invzsp2 = 1.0 / zsp2;

                float usp2 = fx2 * xsp2 * invzsp2 + cx2;
                float vsp2 = fy2 * ysp2 * invzsp2 + cy2;
                float errXsp2 = usp2 - kl2.startPointX;
                float errYsp2 = vsp2 - kl2.startPointY;
                if ((errXsp2 * errXsp2 + errYsp2 * errYsp2) > 5.991 * sigmaSquare2)
                    continue;

                const float xep2 = Rcw2.row(0).dot(ep3Dt) + tcw2.at<float>(0);
                const float yep2 = Rcw2.row(1).dot(ep3Dt) + tcw2.at<float>(1);
                const float invzep2 = 1.0 / zep2;

                float uep2 = fx2 * xep2 * invzep2 + cx2;
                float vep2 = fy2 * yep2 * invzep2 + cy2;
                float errXep2 = uep2 - kl2.endPointX;
                float errYep2 = vep2 - kl2.endPointY;
                if ((errXep2 * errXep2 + errYep2 * errYep2) > 5.991 * sigmaSquare2)
                    continue;

                //Check scale consistency
                cv::Mat normalsp1 = sp3D - Ow1;
                float distsp1 = cv::norm(normalsp1);

                cv::Mat normalep1 = ep3D - Ow1;
                float distep1 = cv::norm(normalep1);

                cv::Mat normalsp2 = sp3D - Ow2;
                float distsp2 = cv::norm(normalsp2);

                cv::Mat normalep2 = ep3D - Ow2;
                float distep2 = cv::norm(normalep2);

                if (distsp1 == 0 || distep1 == 0 || distsp2 == 0 || distep2 == 0)
                    continue;

                const float ratioDistsp = distsp2 / distsp1;
                const float ratioDistep = distep2 / distep1;
                const float ratioOctave =
                        mpCurrentKeyFrame->mvScaleFactors[kl1.octave] / pKF2->mvScaleFactors[kl2.octave];

                if (ratioDistsp * ratioFactor < ratioOctave || ratioDistsp > ratioOctave * ratioFactor ||
                    ratioDistep * ratioFactor < ratioOctave || ratioDistep > ratioOctave * ratioFactor)
                    continue;

                Vector6d line3D;
                line3D << sp3Dt.at<float>(0), sp3Dt.at<float>(1), sp3Dt.at<float>(2), ep3Dt.at<float>(0),
                        ep3Dt.at<float>(1), ep3Dt.at<float>(2);
                int manh_idx = mpCurrentKeyFrame->vManhAxisIdx[idx1];
                MapLine *pML = new MapLine(line3D, manh_idx, mpCurrentKeyFrame, mpMap);

                pML->AddObservation(mpCurrentKeyFrame, idx1);
                pML->AddObservation(pKF2, idx2);

                mpCurrentKeyFrame->AddMapLine(pML, idx1);
                pKF2->AddMapLine(pML, idx2);

                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();
                mpMap->AddMapLine(pML);

                mlpRecentAddedMapLines.push_back(pML);

                nnew++;
            }

        }
    }

void LocalMapping::CreateNewMapLines()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn=5;
    if(mbMonocular)
        nn=10;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    
    if(vpNeighKFs.size() < 3)
        return;

    LSDmatcher lmatcher(0.75);    

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const Mat &K1 = mpCurrentKeyFrame->mK;
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactorLine;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>0 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2 - Ow1;
        // 基线长度
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        vector<pair<size_t, size_t>> vMatchedIndices;
        lmatcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices);

        ////////////////////////////////////////////////////////////////////////////////////////////
        vector<double> offsets;
        for(int ipair = 0; ipair<vMatchedIndices.size();ipair++)
        {
            KeyLine kl1 = mpCurrentKeyFrame->mvKeyLines[vMatchedIndices[ipair].first];
            KeyLine kl2 = pKF2->mvKeyLines[vMatchedIndices[ipair].second];
            double midX1 = (kl1.startPointX + kl1.endPointX)/2; double midY1 = (kl1.startPointY + kl1.endPointY)/2;
            double midX2 = (kl2.startPointX + kl2.endPointX)/2; double midY2 = (kl2.startPointY + kl2.endPointY)/2;
            double offset = sqrt((midX1 - midX2) * (midX1 - midX2) + (midY1 - midY2)  * (midY1 - midY2)) ;
            offsets.push_back(offset);
        }

        double sum = std::accumulate(std::begin(offsets), std::end(offsets), 0.0);
        double mean =  sum / offsets.size(); 
    
        double accum  = 0.0;
        std::for_each (std::begin(offsets), std::end(offsets), [&](const double d) {
            accum  += (d-mean)*(d-mean);
        });
    
        double stdev = sqrt(accum/(offsets.size()-1));
        ////////////////////////////////////////////////////////////////////////////////////////////

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const Mat &K2 = pKF2->mK;
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each matched line Segment
        const int nmatches = vMatchedIndices.size();
        for(int ikl=0; ikl<nmatches; ikl++)
        {
            if(offsets[ikl] - mean  > stdev * 3){
                //cerr<<vMatchedIndices[ikl].second<<endl;
                continue;
            }
                
            const int &idx1 = vMatchedIndices[ikl].first;
            const int &idx2 = vMatchedIndices[ikl].second;

            const KeyLine &keyline1 = mpCurrentKeyFrame->mvKeyLines[idx1];
            const KeyLine &keyline2 = pKF2->mvKeyLines[idx2];
            const Vector3d keyline1_function = mpCurrentKeyFrame->mvKeyLineFunctions[idx1];
            const Vector3d keyline2_function = pKF2->mvKeyLineFunctions[idx2];
            const Mat klF1 = (Mat_<float>(3,1) << keyline1_function(0),
                                                keyline1_function(1),
                                                keyline1_function(2));
            const Mat klF2 = (Mat_<float>(3,1) << keyline2_function(0),
                                                keyline2_function(1),
                                                keyline2_function(2));

            cv::Mat lineVector2 = (Mat_<float>(2,1) << -keyline2_function(1), keyline2_function(0));
            cv::Mat _ray1Start = (Mat_<float>(3,1) <<keyline1.startPointX, keyline1.startPointY, 1); cv::Mat _ray1End = (Mat_<float>(3,1) << keyline1.endPointX, keyline1.endPointY, 1);
            cv::Mat R21 = Rcw2 * Rwc1;
            cv::Mat t21 = Rcw2 * ( Rwc2 * tcw2 - Rwc1 * tcw1 );
            cv::Mat t21x = SkewSymmetricMatrix(t21);
            cv::Mat F21 = (K2.t()).inv() * t21x * R21 * K1.inv();
            cv::Mat Th1 = F21*_ray1Start;
            cv::Mat Th1_ = (Mat_<float>(2,1) << -Th1.at<float>(1, 0), Th1.at<float>(0, 0));
            float Result1 =  Th1_.dot(lineVector2) / (norm(Th1_) * norm(lineVector2));
            cv::Mat Th2 = F21*_ray1End;
            cv::Mat Th2_ = (Mat_<float>(2,1) << -Th2.at<float>(1, 0), Th2.at<float>(0, 0));
            float Result2 =  Th2_.dot(lineVector2) / (norm(Th2_) * norm(lineVector2));

            if(abs(Result1)>0.98 || abs(Result2)>0.98)
                continue;

            cv::Mat StartC1, EndC1;
            StartC1 = (cv::Mat_<float>(3,1) << (keyline1.startPointX-cx1)*invfx1, (keyline1.startPointY-cy1)*invfy1, 1.0);
            EndC1 = (cv::Mat_<float>(3,1) << (keyline1.endPointX-cx1)*invfx1, (keyline1.endPointY-cy1)*invfy1, 1.0);

            Mat M1 = K1 * Tcw1;
            Mat M2 = K2 * Tcw2;

            cv::Mat s3D, e3D;
            cv::Mat A(4,4,CV_32F);
            A.row(0) = klF1.t()*M1;
            A.row(1) = klF2.t()*M2;
            A.row(2) = StartC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
            A.row(3) = StartC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);

            cv::Mat w1, u1, vt1;
            cv::SVD::compute(A, w1, u1, vt1, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

            s3D = vt1.row(3).t();

            if(s3D.at<float>(3)==0)
                continue;

            // Euclidean coordinates
            s3D = s3D.rowRange(0,3)/s3D.at<float>(3);

            cv::Mat B(4,4,CV_32F);
            B.row(0) = klF1.t()*M1;
            B.row(1) = klF2.t()*M2;
            B.row(2) = EndC1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
            B.row(3) = EndC1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);

            cv::Mat w2, u2, vt2;
            cv::SVD::compute(B, w2, u2, vt2, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

            e3D = vt2.row(3).t();

            if(e3D.at<float>(3)==0)
                continue;

            // Euclidean coordinates
            e3D = e3D.rowRange(0,3)/e3D.at<float>(3);

            cv::Mat s3Dt = s3D.t();
            cv::Mat e3Dt = e3D.t();

            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            cv::Mat v1 = s3D - Ow1;
            float distance1 = cv::norm(v1);
            const float ratio1 = distance1/medianDepthKF2;
            if(ratio1 < 0.3)
                continue;

            cv::Mat v2 = s3D - Ow2;
            float distance2 = cv::norm(v2);
            const float ratio2 = distance2/medianDepthKF2;
            if(ratio2 < 0.3)
                continue;

            cv::Mat v3 = e3D - s3D;
            float distance3 = cv::norm(v3);
            const float ratio3 = distance3/medianDepthKF2;
            if(ratio3 > 1)
                continue;

            float SZC1 = Rcw1.row(2).dot(s3Dt)+tcw1.at<float>(2);  
            if(SZC1<=0)
                continue;

            float SZC2 = Rcw2.row(2).dot(s3Dt)+tcw2.at<float>(2);   
            if(SZC2<=0)
                continue;

            float EZC1 = Rcw1.row(2).dot(e3Dt)+tcw1.at<float>(2);   
            if(EZC1<=0)
                continue;

            float EZC2 = Rcw2.row(2).dot(e3Dt)+tcw2.at<float>(2);   
            if(EZC2<=0)
                continue;

            Vector6d line3D;
            line3D << s3D.at<float>(0), s3D.at<float>(1), s3D.at<float>(2), e3D.at<float>(0), e3D.at<float>(1), e3D.at<float>(2);
            int manh_idx = mpCurrentKeyFrame->vManhAxisIdx[ikl];
            MapLine* pML = new MapLine(line3D, manh_idx, mpCurrentKeyFrame, mpMap);

            pML->AddObservation(mpCurrentKeyFrame, idx1);
            pML->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapLine(pML, idx1);
            pKF2->AddMapLine(pML, idx2);

            pML->ComputeDistinctiveDescriptors();
            pML->UpdateAverageDir();
            mpMap->AddMapLine(pML);

            mlpRecentAddedMapLines.push_back(pML);

            nnew++;
        }

    }
}

void LocalMapping::CreateNewMapLinesConstraint()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn=5;
    if(mbMonocular)
        nn=10;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    
    if(vpNeighKFs.size() < 2)
        return;

    LSDmatcher lmatcher(0.85);   

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();
    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    const Mat &K1 = mpCurrentKeyFrame->mK;
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactorLine;

    int nnew = 0;

    // Search matches with epipolar restriction and triangulate
    
    vector<vector<int>> TotalvMatchedIndices;
    vector<int> nTotalMatched;
    nTotalMatched.reserve(vpNeighKFs.size());
    TotalvMatchedIndices.reserve(vpNeighKFs.size());

    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        if(i>1 && CheckNewKeyFrames())
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2 - Ow1;
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular)
        {
            if(baseline<pKF2->mb)
                continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;
            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Search matches that fulfill epipolar constraint

        vector<int> vMatchedIndices;
        // int nlmatch = lmatcher.SearchForTriangulationNew(mpCurrentKeyFrame, pKF2, vMatchedIndices, true);
        int nlmatch = lmatcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, vMatchedIndices, true);
        TotalvMatchedIndices.push_back(vMatchedIndices);
        nTotalMatched.push_back(nlmatch);
    }

    if(TotalvMatchedIndices.size() < 2)
        return;

    for(size_t i = 0; i<TotalvMatchedIndices.size() -1; i++)
    {
        vector<int> vMatchedIndices1 = TotalvMatchedIndices[i];

        if(nTotalMatched[i] == 0)
            continue;

        KeyFrame* pKF2 = vpNeighKFs[i];

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        cv::Mat Ow2 = pKF2->GetCameraCenter();

        const Mat &K2 = pKF2->mK;
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        for(size_t j = i+1; j<TotalvMatchedIndices.size(); j++)
        {
            vector<int> vMatchedIndices2 = TotalvMatchedIndices[j];

            if(nTotalMatched[j] == 0)
                continue;

            KeyFrame* pKF3 = vpNeighKFs[j];

            cv::Mat Rcw3 = pKF3->GetRotation();
            cv::Mat Rwc3 = Rcw3.t();
            cv::Mat tcw3 = pKF3->GetTranslation();
            cv::Mat Tcw3(3, 4, CV_32F);
            Rcw3.copyTo(Tcw3.colRange(0,3));
            tcw3.copyTo(Tcw3.col(3));

            cv::Mat Ow3 = pKF3->GetCameraCenter();

            const Mat &K3 = pKF3->mK;
            const float &fx3 = pKF3->fx;
            const float &fy3 = pKF3->fy;
            const float &cx3 = pKF3->cx;
            const float &cy3 = pKF3->cy;
            const float &invfx3 = pKF3->invfx;
            const float &invfy3 = pKF3->invfy;

            for(int ikl=0; ikl<mpCurrentKeyFrame->NL; ikl++)
            {   
                const int &idx1 = vMatchedIndices1[ikl];
                const int &idx2 = vMatchedIndices2[ikl];

                if(idx1 == -1 || idx2 == -1 || idx1>=pKF2->NL || idx2 >= pKF3->NL)
                    continue;

                if(mpCurrentKeyFrame->GetMapLine(ikl) || pKF2->GetMapLine(idx1) || pKF3->GetMapLine(idx2))
                    continue;

                const KeyLine &keyline1 = mpCurrentKeyFrame->mvKeyLines[ikl];
                const KeyLine &keyline2 = pKF2->mvKeyLines[idx1];
                const KeyLine &keyline3 = pKF3->mvKeyLines[idx2];
                const Vector3d keyline1_function = mpCurrentKeyFrame->mvKeyLineFunctions[ikl];
                const Vector3d keyline2_function = pKF2->mvKeyLineFunctions[idx1];
                const Vector3d keyline3_function = pKF3->mvKeyLineFunctions[idx2];
                const Mat klF1 = (Mat_<float>(3,1) << keyline1_function(0),
                                                    keyline1_function(1),
                                                    keyline1_function(2));
                const Mat klF2 = (Mat_<float>(3,1) << keyline2_function(0),
                                                    keyline2_function(1),
                                                    keyline2_function(2));
                const Mat klF3 = (Mat_<float>(3,1) << keyline3_function(0),
                                                    keyline3_function(1),
                                                    keyline3_function(2));

                cv::Mat lineVector2 = (Mat_<float>(2,1) << -keyline2_function(1), keyline2_function(0));
                cv::Mat _ray1Start = (Mat_<float>(3,1) <<keyline1.startPointX, keyline1.startPointY, 1); cv::Mat _ray1End = (Mat_<float>(3,1) << keyline1.endPointX, keyline1.endPointY, 1);
                cv::Mat R21 = Rcw2 * Rwc1;
                cv::Mat t21 = Rcw2 * ( Rwc2 * tcw2 - Rwc1 * tcw1 );
                cv::Mat t21x = SkewSymmetricMatrix(t21);
                cv::Mat F21 = (K2.t()).inv() * t21x * R21 * K1.inv();
                cv::Mat Th1 = F21*_ray1Start;
                cv::Mat Th1_ = (Mat_<float>(2,1) << -Th1.at<float>(1, 0), Th1.at<float>(0, 0));
                float Result1 =  Th1_.dot(lineVector2) / (norm(Th1_) * norm(lineVector2));
                cv::Mat Th2 = F21*_ray1End;
                cv::Mat Th2_ = (Mat_<float>(2,1) << -Th2.at<float>(1, 0), Th2.at<float>(0, 0));
                float Result2 =  Th2_.dot(lineVector2) / (norm(Th2_) * norm(lineVector2));

                if(abs(Result1)>0.996 || abs(Result2)>0.996)
                    continue;

                cv::Mat R12 = Rcw1*Rwc2;
                cv::Mat R13 = Rcw1*Rwc3;

                cv::Mat lS = (cv::Mat_<float>(3, 1) << keyline1.startPointX, keyline1.startPointY, 1.0);
                cv::Mat lE = (cv::Mat_<float>(3, 1) << keyline1.endPointX, keyline1.endPointY, 1.0);
                cv::Mat lS_ = K1.inv() * lS; cv::Mat lE_ = K1.inv() * lE;
                cv::Mat L1 = lS_.cross(lE_);

                lS = (cv::Mat_<float>(3, 1) << keyline2.startPointX, keyline2.startPointY, 1.0);
                lE = (cv::Mat_<float>(3, 1) << keyline2.endPointX, keyline2.endPointY, 1.0);
                lS_ = K2.inv() * lS; lE_ = K2.inv() * lE;
                cv::Mat L2 = lS_.cross(lE_);

                lS = (cv::Mat_<float>(3, 1) << keyline3.startPointX, keyline3.startPointY, 1.0);
                lE = (cv::Mat_<float>(3, 1) << keyline3.endPointX, keyline3.endPointY, 1.0);
                lS_ = K3.inv() * lS; lE_ = K3.inv() * lE;
                cv::Mat L3 = lS_.cross(lE_);

                cv::Mat tWorldVector = (R12*L2).cross(R13*L3);
                float norm_ = cv::norm(tWorldVector);
                if(norm_ == 0)
                    continue;
                tWorldVector /= norm_;
                norm_ = cv::norm(L1); L1 /= norm_;
                if(norm_ == 0)
                    continue;
                float CosSita = abs(L1.dot(tWorldVector));

                if(CosSita>0.0087)
                {
                    continue;
                }

                cv::Mat StartC1, EndC1, StartC2, EndC2, StartC3, EndC3;
                StartC1 = (cv::Mat_<float>(3,1) << (keyline1.startPointX-cx1)*invfx1, (keyline1.startPointY-cy1)*invfy1, 1.0);
                EndC1 = (cv::Mat_<float>(3,1) << (keyline1.endPointX-cx1)*invfx1, (keyline1.endPointY-cy1)*invfy1, 1.0);

                cv::Mat M1 = K1 * Tcw1;
                cv::Mat M2 = K2 * Tcw2;
                cv::Mat M3 = K3 * Tcw3;

                cv::Mat s3D, e3D;
                cv::Mat A(4,4,CV_32F);
                A.row(0) = klF3.t()*M3;
                A.row(1) = klF2.t()*M2;
                A.row(2) = keyline1.startPointX*M1.row(2)-M1.row(0);
                A.row(3) = keyline1.startPointY*M1.row(2)-M1.row(1);

                cv::Mat w1, u1, vt1;
                cv::SVD::compute(A, w1, u1, vt1, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                s3D = vt1.row(3).t();

                if(s3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                s3D = s3D.rowRange(0,3)/s3D.at<float>(3);

                // 终止点
                cv::Mat B(4,4,CV_32F);
                B.row(0) = klF3.t()*M3;
                B.row(1) = klF2.t()*M2;
                B.row(2) = keyline1.endPointX*M1.row(2)-M1.row(0);
                B.row(3) = keyline1.endPointY*M1.row(2)-M1.row(1);

                cv::Mat w2, u2, vt2;
                cv::SVD::compute(B, w2, u2, vt2, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                e3D = vt2.row(3).t();

                if(e3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                e3D = e3D.rowRange(0,3)/e3D.at<float>(3);

                cv::Mat s3Dt = s3D.t();
                cv::Mat e3Dt = e3D.t();

                cv::Mat normal1 = s3D - Ow1;
                float dist1 = cv::norm(normal1);

                cv::Mat normal2 = s3D - Ow2;
                float dist2 = cv::norm(normal2);

                cv::Mat normal3 = s3D - Ow3;
                float dist3 = cv::norm(normal3);

                float cosParallax1 = normal1.dot(normal2)/(dist1*dist2);
                float cosParallax2 = normal1.dot(normal3)/(dist1*dist3);

                if(cosParallax1 >= 0.99998 || cosParallax2 >= 0.99998)
                    continue;

                normal1 = e3D - Ow1;
                dist1 = cv::norm(normal1);

                normal2 = e3D - Ow2;
                dist2 = cv::norm(normal2);

                normal3 = e3D - Ow3;
                dist3 = cv::norm(normal3);

                cosParallax1 = normal1.dot(normal2)/(dist1*dist2);
                cosParallax2 = normal1.dot(normal3)/(dist1*dist3);

                if(cosParallax1 >= 0.99998 || cosParallax2 >= 0.99998)
                    continue;

                const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
                cv::Mat v1 = s3D - Ow1;
                float distance1 = cv::norm(v1);
                const float ratio1 = distance1/medianDepthKF2;
                if(ratio1 < 0.3)
                    continue;

                cv::Mat v2 = s3D - Ow2;
                float distance2 = cv::norm(v2);
                const float ratio2 = distance2/medianDepthKF2;
                if(ratio2 < 0.3)
                    continue;

                cv::Mat v3 = e3D - s3D;
                float distance3 = cv::norm(v3);
                const float ratio3 = distance3/medianDepthKF2;
                if(ratio3 > 1)
                    continue;

                float SZC1 = Rcw1.row(2).dot(s3Dt)+tcw1.at<float>(2);   
                if(SZC1<=0)
                    continue;

                float EZC1 = Rcw1.row(2).dot(e3Dt)+tcw1.at<float>(2);   
                if(EZC1<=0)
                    continue;

                float SZC2 = Rcw2.row(2).dot(s3Dt)+tcw2.at<float>(2);   
                if(SZC2<=0)
                    continue;

                float EZC2 = Rcw2.row(2).dot(e3Dt)+tcw2.at<float>(2);   
                if(EZC2<=0)
                    continue;

                float SZC3 = Rcw3.row(2).dot(s3Dt)+tcw3.at<float>(2);   
                if(SZC3<=0)
                    continue;

                float EZC3 = Rcw3.row(2).dot(e3Dt)+tcw3.at<float>(2);   
                if(EZC3<=0)
                    continue;

               
                const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2Line[keyline1.octave];
                const float x1s = Rcw1.row(0).dot(s3Dt)+tcw1.at<float>(0);
                const float y1s = Rcw1.row(1).dot(s3Dt)+tcw1.at<float>(1);
                const float invz1s = 1.0/SZC1;

                float u1s = fx1*x1s*invz1s+cx1;
                float v1s = fy1*y1s*invz1s+cy1;
                double err1s = keyline1_function(0)*u1s + keyline1_function(1)*v1s + keyline1_function(2);
                if((err1s*err1s)>3.84*sigmaSquare1)
                    continue;

                const float x1e = Rcw1.row(0).dot(e3Dt)+tcw1.at<float>(0);
                const float y1e = Rcw1.row(1).dot(e3Dt)+tcw1.at<float>(1);
                const float invz1e = 1.0/EZC1;

                float u1e = fx1*x1e*invz1e+cx1;
                float v1e = fy1*y1e*invz1e+cy1;

                double err1e = keyline1_function(0)*u1e + keyline1_function(1)*v1e + keyline1_function(2);
                if((err1e*err1e)>3.84*sigmaSquare1)
                    continue;

                const float &sigmaSquare2 = pKF2->mvLevelSigma2Line[keyline2.octave];

                const float x2s = Rcw2.row(0).dot(s3Dt)+tcw2.at<float>(0);
                const float y2s = Rcw2.row(1).dot(s3Dt)+tcw2.at<float>(1);
                const float invz2s = 1.0/SZC2;

                float u2s = fx2*x2s*invz2s+cx2;
                float v2s = fy2*y2s*invz2s+cy2;
                double err2s = keyline2_function(0)*u2s + keyline2_function(1)*v2s + keyline2_function(2);
                if((err2s*err2s)>3.84*sigmaSquare2)
                    continue;
                
                const float x2e = Rcw2.row(0).dot(e3Dt)+tcw2.at<float>(0);
                const float y2e = Rcw2.row(1).dot(e3Dt)+tcw2.at<float>(1);
                const float invz2e = 1.0/EZC2;

                float u2e = fx2*x2e*invz2e+cx2;
                float v2e = fy2*y2e*invz2e+cy2;

                double err2e = keyline2_function(0)*u2e + keyline2_function(1)*v2e + keyline2_function(2);
                if((err2e*err2e)>3.84*sigmaSquare2)
                    continue;

                const float &sigmaSquare3 = pKF3->mvLevelSigma2Line[keyline3.octave];

                const float x3s = Rcw3.row(0).dot(s3Dt)+tcw3.at<float>(0);
                const float y3s = Rcw3.row(1).dot(s3Dt)+tcw3.at<float>(1);
                const float invz3s = 1.0/SZC3;

                float u3s = fx3*x3s*invz3s+cx3;
                float v3s = fy3*y3s*invz3s+cy3;

                double err3s = keyline3_function(0)*u3s + keyline3_function(1)*v3s + keyline3_function(2);
                if((err3s*err3s)>3.84*sigmaSquare3)
                    continue;
                
                const float x3e = Rcw3.row(0).dot(e3Dt)+tcw3.at<float>(0);
                const float y3e = Rcw3.row(1).dot(e3Dt)+tcw3.at<float>(1);
                const float invz3e = 1.0/EZC3;

                float u3e = fx3*x3e*invz3e+cx3;
                float v3e = fy3*y3e*invz3e+cy3;

                double err3e = keyline3_function(0)*u3e + keyline3_function(1)*v3e + keyline3_function(2);
                if((err3e*err3e)>3.84*sigmaSquare3)
                    continue;

                if(fabs(keyline1.angle) < 3.0*PI/4.0 && fabs(keyline1.angle) > 1.0*PI/4.0)
                {
                    if(min(v1e, v1s) > max(keyline1.startPointY, keyline1.endPointY) || min(keyline1.startPointY, keyline1.endPointY) > max(v1e, v1s))
                        continue;

                    float Ymax = min(max(v1e, v1s), max(keyline1.startPointY, keyline1.endPointY));
                    float Ymin= max(min(v1e, v1s), min(keyline1.startPointY, keyline1.endPointY));

                    float ratio1 = (Ymax - Ymin)/(max(v1e, v1s)-min(v1e, v1s));
                    float ratio2 = (Ymax - Ymin)/(max(keyline1.startPointY, keyline1.endPointY)-min(keyline1.startPointY, keyline1.endPointY));

                    if(ratio1 < 0.85 || ratio2 < 0.85)
                        continue;
                }else{
                     if(min(u1e, u1s) > max(keyline1.startPointX, keyline1.endPointX) || min(keyline1.startPointX, keyline1.endPointX) > max(u1e, u1s))
                        continue;

                    float Xmax = min(max(u1e, u1s), max(keyline1.startPointX, keyline1.endPointX));
                    float Xmin= max(min(u1e, u1s), min(keyline1.startPointX, keyline1.endPointX));

                    float ratio1 = (Xmax - Xmin)/(max(u1e, u1s)-min(u1e, u1s));
                    float ratio2 = (Xmax - Xmin)/(max(keyline1.startPointX, keyline1.endPointX)-min(keyline1.startPointX, keyline1.endPointX));

                    if(ratio1 < 0.85 || ratio2 < 0.85)
                        continue;
                }

                if(fabs(keyline2.angle) < 3.0*PI/4.0 && fabs(keyline2.angle) > 1.0*PI/4.0)
                {
                    if(min(v2e, v2s) > max(keyline2.startPointY, keyline2.endPointY) || min(keyline2.startPointY, keyline2.endPointY) > max(v2e, v2s))
                        continue;

                    float Ymax = min(max(v2e, v2s), max(keyline2.startPointY, keyline2.endPointY));
                    float Ymin= max(min(v2e, v2s), min(keyline2.startPointY, keyline2.endPointY));

                    float ratio1 = (Ymax - Ymin)/(max(v2e, v2s)-min(v2e, v2s));
                    float ratio2 = (Ymax - Ymin)/(max(keyline2.startPointY, keyline2.endPointY)-min(keyline2.startPointY, keyline2.endPointY));

                    if(ratio1 < 0.85 || ratio2 < 0.85)
                        continue;
                }else{
                     if(min(u2e, u2s) > max(keyline2.startPointX, keyline2.endPointX) || min(keyline2.startPointX, keyline2.endPointX) > max(u2e, u2s))
                        continue;

                    float Xmax = min(max(u2e, u2s), max(keyline2.startPointX, keyline2.endPointX));
                    float Xmin= max(min(u2e, u2s), min(keyline2.startPointX, keyline2.endPointX));

                    float ratio1 = (Xmax - Xmin)/(max(u2e, u2s)-min(u2e, u2s));
                    float ratio2 = (Xmax - Xmin)/(max(keyline2.startPointX, keyline2.endPointX)-min(keyline2.startPointX, keyline2.endPointX));

                    if(ratio1 < 0.85 || ratio2 < 0.85)
                        continue;
                }

                if(fabs(keyline3.angle) < 3.0*PI/4.0 && fabs(keyline3.angle) > 1.0*PI/4.0)
                {
                    if(min(v3e, v3s) > max(keyline3.startPointY, keyline3.endPointY) || min(keyline3.startPointY, keyline3.endPointY) > max(v3e, v3s))
                        continue;

                    float Ymax = min(max(v3e, v3s), max(keyline3.startPointY, keyline3.endPointY));
                    float Ymin= max(min(v3e, v3s), min(keyline3.startPointY, keyline3.endPointY));

                    float ratio1 = (Ymax - Ymin)/(max(v3e, v3s)-min(v3e, v3s));
                    float ratio2 = (Ymax - Ymin)/(max(keyline3.startPointY, keyline3.endPointY)-min(keyline3.startPointY, keyline3.endPointY));

                    if(ratio1 < 0.85 || ratio2 < 0.85)
                        continue;
                }else{
                     if(min(u3e, u3s) > max(keyline3.startPointX, keyline3.endPointX) || min(keyline3.startPointX, keyline3.endPointX) > max(u3e, u3s))
                        continue;

                    float Xmax = min(max(u3e, u3s), max(keyline3.startPointX, keyline3.endPointX));
                    float Xmin= max(min(u3e, u3s), min(keyline3.startPointX, keyline3.endPointX));

                    float ratio1 = (Xmax - Xmin)/(max(u3e, u3s)-min(u3e, u3s));
                    float ratio2 = (Xmax - Xmin)/(max(keyline3.startPointX, keyline3.endPointX)-min(keyline3.startPointX, keyline3.endPointX));

                    if(ratio1 < 0.85 || ratio2 < 0.85)
                        continue;
                }

                Vector6d line3D;
                line3D << s3D.at<float>(0), s3D.at<float>(1), s3D.at<float>(2), e3D.at<float>(0), e3D.at<float>(1), e3D.at<float>(2);
                int manh_idx = mpCurrentKeyFrame->vManhAxisIdx[ikl];
                MapLine* pML = new MapLine(line3D, manh_idx, mpCurrentKeyFrame, mpMap);

                pML->AddObservation(mpCurrentKeyFrame, ikl);
                pML->AddObservation(pKF2, idx1);
                pML->AddObservation(pKF3, idx2);

                mpCurrentKeyFrame->AddMapLine(pML, ikl);
                pKF2->AddMapLine(pML, idx1);
                pKF3->AddMapLine(pML, idx2);

                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();
                mpMap->AddMapLine(pML);
                mlpRecentAddedMapLines.push_back(pML);

                nnew++;
            }

        }
    }

}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);    
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    //=====================MapPoint====================
    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    //=====================MapLine====================
    // Search matches by projection from current KF in target KFs
    LSDmatcher lineMatcher(0.75);
    vector<MapLine*> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();     //也就是当前帧的mvpMapLines
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        lineMatcher.Fuse(pKFi, vpMapLineMatches);
    }

    vector<MapLine*> vpLineFuseCandidates;
    vpLineFuseCandidates.reserve(vpTargetKFs.size()*vpMapLineMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapLine*> vpMapLinesKFi = pKFi->GetMapLineMatches();

        for(vector<MapLine*>::iterator vitML=vpMapLinesKFi.begin(), vendML=vpMapLinesKFi.end(); vitML!=vendML; vitML++)
        {
            MapLine* pML = *vitML;
            if(!pML)
                continue;

            if(pML->isBad() || pML->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;

            pML->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpLineFuseCandidates.push_back(pML);
        }
    }

    lineMatcher.Fuse(mpCurrentKeyFrame, vpLineFuseCandidates);

    // Update Lines
    vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();
    for(size_t i=0, iend=vpMapLineMatches.size(); i<iend; i++)
    {
        MapLine* pML=vpMapLineMatches[i];
        if(pML)
        {
            if(!pML->isBad())
            {
                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}


void LocalMapping::SearchLineInNeighbors()
{
    int nn=10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF==mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vit2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId  || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }

    LSDmatcher matcher;

    vector<MapLine*> vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();     //也就是当前帧的mvpMapLines

    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        matcher.Fuse(pKFi, vpMapLineMatches);
    }

    vector<MapLine*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapLineMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapLine*> vpMapLinesKFi = pKFi->GetMapLineMatches();

        for(vector<MapLine*>::iterator vitML=vpMapLinesKFi.begin(), vendML=vpMapLinesKFi.end(); vitML!=vendML; vitML++)
        {
            MapLine* pML = *vitML;
            if(!pML)
                continue;

            if(pML->isBad() || pML->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;

            pML->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pML);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, vpFuseCandidates);

    // Update Lines
    vpMapLineMatches = mpCurrentKeyFrame->GetMapLineMatches();
    for(size_t i=0, iend=vpMapLineMatches.size(); i<iend; i++)
    {
        MapLine* pML=vpMapLineMatches[i];
        if(pML)
        {
            if(!pML->isBad())
            {
                pML->ComputeDistinctiveDescriptors();
                pML->UpdateAverageDir();
            }
        }
    }

    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}

void LocalMapping::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
    unique_lock<mutex> lock2(mMutexNewKFs);
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void LocalMapping::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    unique_lock<mutex> lock2(mMutexFinish);
    if(mbFinished)
        return;
    mbStopped = false;
    mbStopRequested = false;
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit;
    mlNewKeyFrames.clear();

    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept);
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept);
    mbAcceptKeyFrames=flag;
}

bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop);

    if(flag && mbStopped)
        return false;

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

void LocalMapping::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);
            if(!mbResetRequested)
                break;
        }
        usleep(3000);
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();    
        mlpRecentAddedMapLines.clear();    
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
