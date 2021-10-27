//
// Created by lan on 17-12-20.
//

#include "MapLine.h"

#include <mutex>
#include <include/LSDmatcher.h>
#include <map>

namespace ORB_SLAM2
{
    mutex MapLine::mGlobalMutex;
    long unsigned int MapLine::nNextId=0;

MapLine::MapLine(Vector6d &Pos, int &ManhIdx, KeyFrame *pRefKF, Map *pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnBALocalForKFManh(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapLine*>(NULL)), mpMap(pMap)
{
    mWorldPos = Pos;
    ManhAxes = ManhIdx;
    mWorldVector = Pos.head(3) - Pos.tail(3);
    mWorldVector.normalize();

    mNormalVector << 0, 0, 0;

    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId = nNextId++;
}
MapLine::MapLine(Vector6d &Pos, int &ManhIdx, Map *pMap, Frame *pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnBALocalForKFManh(0), mnFuseCandidateForKF(0), mnLoopLineForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    mWorldPos = Pos;
    ManhAxes = ManhIdx;
    mWorldVector = Pos.head(3) - Pos.tail(3);
    mWorldVector.normalize();
    Mat Ow = pFrame->GetCameraCenter(); 
    Vector3d OW;
    OW << Ow.at<double>(0), Ow.at<double>(1), Ow.at<double>(2);
    Eigen::Vector3d start3D = Pos.head(3);
    Eigen::Vector3d end3D = Pos.tail(3);
    Vector3d midPoint = 0.5*(start3D + end3D);
    mNormalVector = midPoint - OW; 
    mNormalVector.normalize();

    Vector3d PC = midPoint - OW;
    const float dist = PC.norm();   

    const int level = pFrame->mvKeylinesUn[idxF].octave;
    const float levelScaleFactor = pFrame->mvScaleFactorsLine[level];
    const int nLevels = pFrame->mnScaleLevelsLine;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactorsLine[nLevels-1];

    pFrame->mLdesc.row(idxF).copyTo(mLDescriptor);

    unique_lock<mutex> lock(mpMap->mMutexLineCreation);
    mnId = nNextId++;
}

    void MapLine::SetWorldPos(const Vector6d &Pos)
    {
        unique_lock<mutex> lock2(mGlobalMutex);
        unique_lock<mutex> lock(mMutexPos);
        mWorldPos = Pos;
        mWorldVector = Pos.head(3) - Pos.tail(3);
        mWorldVector.normalize();
    }

    Vector6d MapLine::GetWorldPos()
    {
        return mWorldPos;
    }

    Vector3d MapLine::GetWorldVector()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mWorldVector;
    }

    int MapLine::GetManhIdx()
    {
        unique_lock<mutex> lock(mMutexPos);
        return ManhAxes;
    }

    Vector3d MapLine::GetNormal()
    {
        unique_lock<mutex> lock(mMutexPos);
        return mNormalVector;
    }

    std::map<KeyFrame*, std::vector<int>> MapLine::GetPerpObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mPerpObservations;
    }

    std::map<KeyFrame*, std::vector<int>> MapLine::GetParObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mParObservations;
    }

    void MapLine::EraseParObs(KeyFrame *pKF, const int &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        if (mParObservations.count(pKF))
        {
            std::vector<int> v_idx = mParObservations[pKF];
            v_idx[idx] = -1;
            mParObservations[pKF] = v_idx;
        }
    }

    void MapLine::ErasePerpObs(KeyFrame *pKF, const int &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        if (mPerpObservations.count(pKF))
        {
            std::vector<int> v_idx = mPerpObservations[pKF];
            v_idx[idx] = -1;
            mPerpObservations[pKF] = v_idx;
        }
    }

    KeyFrame* MapLine::GetReferenceKeyFrame()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    void MapLine::AddObservation(KeyFrame *pKF, size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return;
        mObservations[pKF]=idx;

        nObs++;   
    }

    void MapLine::AddParObservation(KeyFrame *pKF, std::vector<int> idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mParObservations.count(pKF))
            return;
        mParObservations[pKF] = idx;
    }

    void MapLine::AddPerpObservation(KeyFrame *pKF, std::vector<int> idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if (mPerpObservations.count(pKF))
            return;
        mPerpObservations[pKF] = idx;
    }

    void MapLine::EraseObservation(KeyFrame *pKF)
    {
        bool bBad = false;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            if(mObservations.count(pKF))
            {
                int idx = mObservations[pKF];
                if(pKF->mvuRight[idx]>=0)
                    nObs-=2;
                else
                    nObs--;

                mObservations.erase(pKF);

                if(mpRefKF==pKF)
                    mpRefKF=mObservations.begin()->first;

                if(nObs<=2)
                    bBad=true;
            }
        }

        if(bBad)
            SetBadFlag();
    }

     void MapLine::EraseManhObs(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        if (mObservations.count(pKF))
        {
            int v_idx = mObservations[pKF];
            if (v_idx < pKF->vManhAxisIdx.size())
                pKF->vManhAxisIdx[v_idx] = -1;
        }
    }

    map<KeyFrame*, size_t> MapLine::GetObservations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mObservations;
    }

    int MapLine::Observations()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return nObs;
    }

    int MapLine::GetIndexInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
            return mObservations[pKF];
        else
            return -1;
    }

    bool MapLine::IsInKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return (mObservations.count(pKF));
    }

    void MapLine::SetBadFlag()
    {
        map<KeyFrame*, size_t> obs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            mbBad=true;
            obs = mObservations;    
            mObservations.clear();  
        }

        for(map<KeyFrame*, size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;
            pKF->EraseMapLineMatch(mit->second);    
        }

        mpMap->EraseMapLine(this); 
    }

   
    bool MapLine::isBad()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mbBad;
    }

    void MapLine::Replace(MapLine *pML)
    {
        if(pML->mnId==this->mnId)
            return;

        int nvisible, nfound;
        map<KeyFrame*, size_t> obs, verObs, parObs;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            obs=mObservations;
            mObservations.clear();

            mPerpObservations.clear();
            mParObservations.clear();

            mbBad=true;
            nvisible=mnVisible;
            nfound = mnFound;
            mpReplaced = pML;
        }

  
        for(map<KeyFrame*, size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;

            if(!pML->IsInKeyFrame(pKF))
            {
                pKF->ReplaceMapLineMatch(mit->second, pML);
                pML->AddObservation(pKF, mit->second);
            } else{
                pKF->EraseMapLineMatch(mit->second);
            }
        }



        pML->IncreaseFound(nfound);
        pML->IncreaseVisible(nvisible);
        pML->ComputeDistinctiveDescriptors();

        mpMap->EraseMapLine(this);
    }

    MapLine* MapLine::GetReplaced()
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void MapLine::IncreaseVisible(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnVisible+=n;
    }

    void MapLine::IncreaseFound(int n)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mnFound+=n;
    }

    float MapLine::GetFoundRatio()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return static_cast<float>(mnFound)/mnVisible;
    }

    void MapLine::ComputeDistinctiveDescriptors()
    {
        // Retrieve all observed descriptors
        vector<Mat> vDescriptors;

        map<KeyFrame*, size_t> observations;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            if(mbBad)
                return ;
            observations = mObservations;
        }

        if(observations.empty())
            return;

        vDescriptors.reserve(observations.size());

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;

            if(!pKF->isBad())
                vDescriptors.push_back(pKF->mLineDescriptors.row(mit->second));
        }

        if(vDescriptors.empty())
            return;

        const size_t NL=vDescriptors.size();

        std::vector<std::vector<float>> Distances;
        Distances.resize(NL, vector<float>(NL, 0));
        for(size_t i=0; i<NL; i++)
        {
            Distances[i][i]=0;
            for(size_t j=0; j<NL; j++)
            {
                int distij = norm(vDescriptors[i], vDescriptors[j], NORM_HAMMING);
                Distances[i][j]=distij;
                Distances[j][i]=distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx = 0;
        for(size_t i=0; i<NL; i++)
        {
            vector<int> vDists(Distances[i].begin(), Distances[i].end());
            sort(vDists.begin(), vDists.end());

            int median = vDists[0.5*(NL-1)];

            if(median<BestMedian)
            {
                BestMedian = median;
                BestIdx = i;
            }
        }
        {
            unique_lock<mutex> lock(mMutexFeatures);
            mLDescriptor = vDescriptors[BestIdx].clone();
        }

    }

    Mat MapLine::GetDescriptor()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mLDescriptor.clone();
    }

    void MapLine::UpdateAverageDir()
    {
        map<KeyFrame*, size_t> observations;
        KeyFrame* pRefKF;
        Vector6d Pos;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            if(mbBad)
                return;
            observations = mObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos;
        }

        if(observations.empty())
            return;

        Vector3d normal(0, 0, 0);
        int n=0;
        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            KeyFrame* pKF = mit->first;
            Mat Owi = pKF->GetCameraCenter();
            Vector3d OWi(Owi.at<float>(0), Owi.at<float>(1), Owi.at<float>(2));
            Vector3d middlePos = 0.5*(mWorldPos.head(3)+mWorldPos.tail(3));
            Vector3d normali = middlePos - OWi;
            normal = normal + normali/normali.norm();
            n++;
        }

        cv::Mat SP = (Mat_<float>(3,1) << Pos(0), Pos(1), Pos(2));
        cv::Mat EP = (Mat_<float>(3,1) << Pos(3), Pos(4), Pos(5));
        cv::Mat MP = 0.5*(SP+EP);

        cv::Mat CM = MP - pRefKF->GetCameraCenter();
        const float dist = cv::norm(CM);
        const int level = pRefKF->mvKeyLines[observations[pRefKF]].octave;
        const float levelScaleFactor = pRefKF->mvScaleFactorsLine[level];
        const int nLevels = pRefKF->mnScaleLevelsLine;


        {
            unique_lock<mutex> lock3(mMutexPos);
            mfMaxDistance = dist*levelScaleFactor;
            mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactorsLine[nLevels-1];
            mNormalVector = normal/n;

            mWorldVector = Pos.head(3) - Pos.tail(3);
            mWorldVector.normalize();
        }
    }

    void MapLine::UpdateManhAxis()
    {
          {
            unique_lock<mutex> lock3(mMutexPos);
            cv::Mat manh_axis = mpMap->GetWorldManhAxis();
            if(manh_axis.empty())
            {
                ManhAxes = 0;
                return;
            }
           
            // Check Manh Axis
            if (ManhAxes == 0)
            {
                double th_angle = 10;
                double th_rad = th_angle / 180.0 * M_PI;
                double th_normal = std::cos(th_rad);

                cv::Mat m_normal = (Mat_<double>(3, 1) << mWorldVector[0],
                                    mWorldVector[1],
                                    mWorldVector[2]);

                 Manhattan *mpManh;
                 for (size_t i = 0; i < 3; i++)
                 {
                     double angle = mpManh->computeAngle(m_normal, manh_axis.col(i));

                     if (angle > th_normal)
                     {
                         ManhAxes = i + 1;
                         break;
                     }
                }
            }
        }
    }

    // TODO: Modify it
    void MapLine::UpdateNormalAndDepth()
    {
        map<KeyFrame *, size_t> observations;
        KeyFrame *pRefKF;
        Vector6d Pos;
        {
            unique_lock<mutex> lock1(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPos);
            if (mbBad)
                return;
            observations = mObservations;
            pRefKF = mpRefKF;
            Pos = mWorldPos;
        }

        if (observations.empty())
            return;

        cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);

        Eigen::Vector3d st_pt_mean;
        Eigen::Vector3d end_pt_mean;
        int n = 0;
        for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            std::pair<Eigen::Vector3d, Eigen::Vector3d> line_obs = pKF->mvLines3D[mit->second];

            st_pt_mean += line_obs.first;
            end_pt_mean += line_obs.second;
            n++;
        }

        {
            unique_lock<mutex> lock3(mMutexPos);
            // mfMaxDistance = dist * levelScaleFactor;
            // mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
            // mNormalVector = normal / n;
            std::pair<Eigen::Vector3d, Eigen::Vector3d> mean_line(st_pt_mean/n, end_pt_mean/n);
        }
    }

    float MapLine::GetMinDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 0.8f*mfMinDistance;
    }

    float MapLine::GetMaxDistanceInvariance()
    {
        unique_lock<mutex> lock(mMutexPos);
        return 1.2f*mfMaxDistance;
    }

    int MapLine::PredictScale(const float &currentDist, const float &logScaleFactor)
    {
        float ratio;
        {
            unique_lock<mutex> lock3(mMutexPos);
            ratio = mfMaxDistance/currentDist;
        }

        return ceil(log(ratio)/logScaleFactor);
    }
}