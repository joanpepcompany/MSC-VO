//
// Created by lan on 17-12-20.
//

#ifndef ORB_SLAM2_MAPLINE_H
#define ORB_SLAM2_MAPLINE_H

#include "KeyFrame.h"
#include "Frame.h"
#include "Map.h"

#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/core/core.hpp>
#include <mutex>
#include <eigen3/Eigen/Core>
#include <map>

using namespace cv::line_descriptor;
using namespace Eigen;

namespace ORB_SLAM2
{
class KeyFrame;
class Map;
class Frame;

typedef Matrix<double,6,1> Vector6d;
class MapLine
{
public:
    MapLine(int idx_, Vector6d line3D_, Mat desc_, int kf_obs_, Vector3d obs_, Vector3d dir_, Vector4d pts_, double sigma2_=1.f);
    ~MapLine(){};

    void addMapLineObervation(Mat desc_, int kf_obs_, Vector3d obs_, Vector3d dir_, Vector4d pts_, double sigma2_=1.f);

    MapLine(Vector6d &Pos, int &ManhIdx, KeyFrame* pRefKF, Map* pMap);   
    MapLine(Vector6d &Pos, int &ManhIdx, Map* pMap, Frame* pFrame, const int &idxF); 
    void SetWorldPos(const Vector6d &Pos);
    Vector6d GetWorldPos();

    int GetManhIdx();

    Vector3d GetWorldVector();
    Vector3d GetNormal();
    KeyFrame* GetReferenceKeyFrame();

    map<KeyFrame*, size_t> GetObservations();
    int Observations();

    void AddObservation(KeyFrame* pKF, size_t idx);
    void EraseObservation(KeyFrame* pKF);

    void EraseManhObs(KeyFrame *pKF);

    void AddParObservation(KeyFrame *pKF, std::vector<int> idx);
    void AddPerpObservation(KeyFrame *pKF, std::vector<int> idx);

    void EraseParObs(KeyFrame *pKF, const int &idx);
    void ErasePerpObs(KeyFrame *pKF, const int &idx);

    std::map<KeyFrame*, std::vector<int>> GetPerpObservations();
    std::map<KeyFrame*, std::vector<int>> GetParObservations();

    int GetIndexInKeyFrame(KeyFrame* pKF);
    bool IsInKeyFrame(KeyFrame* pKF);

    void SetBadFlag();
    bool isBad();

    void Replace(MapLine* pML);
    MapLine* GetReplaced();

    void IncreaseVisible(int n=1);
    void IncreaseFound(int n=1);
    float GetFoundRatio();
    inline int GetFound(){
        return mnFound;
    }

    void ComputeDistinctiveDescriptors();  

    Mat GetDescriptor();

    void UpdateAverageDir();    

    void UpdateManhAxis();

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance();
    float GetMaxDistanceInvariance();
    int PredictScale(const float &currentDist, const float &logScaleFactor);

public:
    long unsigned int mnId; //Global ID for MapLine
    static long unsigned int nNextId;
    const long int mnFirstKFid; 
    const long int mnFirstFrame;   
    int nObs;

    // Variables used by the tracking
    float mTrackProjX1;
    float mTrackProjY1;
    float mTrackProjX2;
    float mTrackProjY2;
    int mnTrackScaleLevel;
    float mTrackViewCos;
  
    bool mbTrackInView;

    long unsigned int mnTrackReferenceForFrame;

    long unsigned int mnLastFrameSeen;

    // Variables used by local mapping
    long unsigned int mnBALocalForKF;
    long unsigned int mnBALocalForKFManh;
    long unsigned int mnFuseCandidateForKF;

    // Variables used by loop closing
    long unsigned int mnLoopLineForKF;
    long unsigned int mnCorrectedByKF;
    long unsigned int mnCorrectedReference;
    Mat mPosGBA;
    long unsigned int mnBAGlobalForKF;

    static std::mutex mGlobalMutex;

public:
    // Position in absolute coordinates
    Vector6d mWorldPos;
    Vector3d mWorldVector;
    
    int ManhAxes;

    // KeyFrames observing the line and associated index in keyframe
    map<KeyFrame*, size_t> mObservations;  

    std::map<KeyFrame*, std::vector<int>> mParObservations;
    std::map<KeyFrame *, std::vector<int>> mPerpObservations;

    Vector3d mNormalVector; 

    Mat mLDescriptor;   

    KeyFrame* mpRefKF;  

    //Tracking counters
    int mnVisible;
    int mnFound;
    // Bad flag , we don't currently erase MapPoint from memory
    bool mbBad;
    MapLine* mpReplaced;

    float mfMinDistance;
    float mfMaxDistance;

    Map* mpMap;

    mutex mMutexPos;
    mutex mMutexFeatures;
};

}


#endif //ORB_SLAM2_MAPLINE_H
