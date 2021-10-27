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

#include "Optimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <Eigen/StdVector>

#include "Converter.h"

#include <mutex>
using namespace g2o;

namespace ORB_SLAM2
{

    void Optimizer::GlobalBundleAdjustemnt(Map *pMap, int nIterations, const bool bWithLineFeature, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
        vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
        vector<MapPoint *> vpMP = pMap->GetAllMapPoints();
        if (bWithLineFeature)
        {
            vector<MapLine *> vpML = pMap->GetAllMapLines();
            cout << "***** GlobalBA with points & lines *****" << endl;
            BundleAdjustment(vpKFs, vpMP, vpML, nIterations, pbStopFlag, nLoopKF, bRobust);
        }
        else
        {
            cout << "***** GlobalBA with only points ***** " << endl;
            BundleAdjustment(vpKFs, vpMP, nIterations, pbStopFlag, nLoopKF, bRobust);
        }
    }

    void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP,
                                     int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
        vector<bool> vbNotIncludedMP;
        vbNotIncludedMP.resize(vpMP.size());

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if (pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        long unsigned int maxKFid = 0;

        // Set KeyFrame vertices
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (pKF->isBad())
                continue;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
            vSE3->setId(pKF->mnId);
            vSE3->setFixed(pKF->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKF->mnId > maxKFid)
                maxKFid = pKF->mnId;
        }

        const float thHuber2D = sqrt(5.99);
        const float thHuber3D = sqrt(7.815);

        // Set MapPoint vertices
        for (size_t i = 0; i < vpMP.size(); i++)
        {
            MapPoint *pMP = vpMP[i];
            if (pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            int nEdges = 0;
            // SET EDGES
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
            {

                KeyFrame *pKF = mit->first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

                if (pKF->mvuRight[mit->second] < 0)
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    if (bRobust)
                    {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;

                    optimizer.addEdge(e);
                }
                else
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKF->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    if (bRobust)
                    {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber3D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;
                    e->bf = pKF->mbf;

                    optimizer.addEdge(e);
                }
            }

            if (nEdges == 0)
            {
                optimizer.removeVertex(vPoint);
                vbNotIncludedMP[i] = true;
            }
            else
            {
                vbNotIncludedMP[i] = false;
            }
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);

        // Recover optimized data
        // Keyframes
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (pKF->isBad())
                continue;
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            if (nLoopKF == 0)
            {
                pKF->SetPose(Converter::toCvMat(SE3quat));
            }
            else
            {
                pKF->mTcwGBA.create(4, 4, CV_32F);
                Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
                pKF->mnBAGlobalForKF = nLoopKF;
            }
        }

        // Points
        for (size_t i = 0; i < vpMP.size(); i++)
        {
            if (vbNotIncludedMP[i])
                continue;

            MapPoint *pMP = vpMP[i];

            if (pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

            if (nLoopKF == 0)
            {
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA.create(3, 1, CV_32F);
                Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }
    }

    void Optimizer::BundleAdjustment(const vector<KeyFrame *> &vpKFs, const vector<MapPoint *> &vpMP, const vector<MapLine *> &vpML,
                                     int nIterations, bool *pbStopFlag, const unsigned long nLoopKF, const bool bRobust)
    {
        double invSigma = 1;
        vector<bool> vbNotIncludedMP;
        vbNotIncludedMP.resize(vpMP.size());

        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if (pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        long unsigned int maxKFid = 0;

        cout << "======= Optimizer::BundleAdjustment with lines =======" << endl;
        // Set KeyFrame vertices
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (pKF->isBad())
                continue;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
            vSE3->setId(pKF->mnId);
            vSE3->setFixed(pKF->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKF->mnId > maxKFid)
                maxKFid = pKF->mnId;
        }

        const float thHuber2D = sqrt(5.99);
        const float thHuber3D = sqrt(7.815);
        const float thHuberLD = sqrt(3.84);

        vector<int> MapPointID;

        // ********************************Set MapPoint vertices*******************************
        for (size_t i = 0; i < vpMP.size(); i++)
        {
            MapPoint *pMP = vpMP[i];
            if (pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            const int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            MapPointID.push_back(id);

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            int nEdges = 0;
            // SET EDGES
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
            {

                KeyFrame *pKF = mit->first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nEdges++;

                const cv::KeyPoint &kpUn = pKF->mvKeysUn[mit->second];

                if (pKF->mvuRight[mit->second] < 0)
                {
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    if (bRobust)
                    {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber2D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;

                    optimizer.addEdge(e);
                }
                else
                {
                    Eigen::Matrix<double, 3, 1> obs;
                    const float kp_ur = pKF->mvuRight[mit->second];
                    obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                    g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                    Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                    e->setInformation(Info);

                    if (bRobust)
                    {
                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuber3D);
                    }

                    e->fx = pKF->fx;
                    e->fy = pKF->fy;
                    e->cx = pKF->cx;
                    e->cy = pKF->cy;
                    e->bf = pKF->mbf;

                    optimizer.addEdge(e);
                }
            }

            if (nEdges == 0)
            {
                optimizer.removeVertex(vPoint);
                vbNotIncludedMP[i] = true;
            }
            else
            {
                vbNotIncludedMP[i] = false;
            }
        }

        std::sort(MapPointID.begin(), MapPointID.end());
        int maxMapPointID = MapPointID[MapPointID.size() - 1];

        for (size_t i = 0; i < vpML.size(); i++)
        {
            MapLine *pML = vpML[i];
            if (pML->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vStartP = new g2o::VertexSBAPointXYZ();
            vStartP->setEstimate(pML->GetWorldPos().head(3));
            const int ids = pML->mnId + maxKFid + maxMapPointID + 1;
            vStartP->setId(ids);
            vStartP->setMarginalized(true);
            optimizer.addVertex(vStartP);

            const map<KeyFrame *, size_t> observations = pML->mObservations;

            int nLineEdges = 0;

            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
            {
                KeyFrame *pKF = mit->first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nLineEdges++;

                Eigen::Vector3d line_obs;
                line_obs = pKF->mvKeyLineFunctions[mit->second];

                DistPt2Line2DMultiFrame *e = new DistPt2Line2DMultiFrame();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ids)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(line_obs);
                e->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                if (bRobust)
                {
                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(thHuberLD);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                // e->Xw = pML->mWorldPos.head(3);

                optimizer.addEdge(e);
            }
        }

        for (size_t i = 0; i < vpML.size(); i++)
        {
            MapLine *pML = vpML[i];
            if (pML->isBad())
                continue;
            g2o::VertexSBAPointXYZ *vEndP = new VertexSBAPointXYZ();
            vEndP->setEstimate(pML->GetWorldPos().tail(3));
            const int ide = pML->mnId + maxKFid + maxMapPointID + vpML.size() + 1;
            vEndP->setId(ide);
            vEndP->setMarginalized(true);
            optimizer.addVertex(vEndP);

            const map<KeyFrame *, size_t> observations = pML->mObservations;

            int nLineEdges = 0;

            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(); mit != observations.end(); mit++)
            {
                KeyFrame *pKF = mit->first;
                if (pKF->isBad() || pKF->mnId > maxKFid)
                    continue;

                nLineEdges++;

                Eigen::Vector3d line_obs;
                line_obs = pKF->mvKeyLineFunctions[mit->second];

                DistPt2Line2DMultiFrame *e = new DistPt2Line2DMultiFrame();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ide)));
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                e->setMeasurement(line_obs);
                e->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                if (bRobust)
                {
                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(thHuberLD);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                // e->Xw = pML->mWorldPos.head(3);

                optimizer.addEdge(e);
            }
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(nIterations);

        // Recover optimized data

        // Keyframes
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (pKF->isBad())
                continue;
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            if (nLoopKF == 0)
            {
                pKF->SetPose(Converter::toCvMat(SE3quat));
            }
            else
            {
                pKF->mTcwGBA.create(4, 4, CV_32F);
                Converter::toCvMat(SE3quat).copyTo(pKF->mTcwGBA);
                pKF->mnBAGlobalForKF = nLoopKF;
            }
        }

        // Points
        for (size_t i = 0; i < vpMP.size(); i++)
        {
            if (vbNotIncludedMP[i])
                continue;

            MapPoint *pMP = vpMP[i];

            if (pMP->isBad())
                continue;
            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));

            if (nLoopKF == 0)
            {
                pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
                pMP->UpdateNormalAndDepth();
            }
            else
            {
                pMP->mPosGBA.create(3, 1, CV_32F);
                Converter::toCvMat(vPoint->estimate()).copyTo(pMP->mPosGBA);
                pMP->mnBAGlobalForKF = nLoopKF;
            }
        }

        // Line  EndPoints
        for (size_t i = 0; i < vpML.size(); i++)
        {
            if (vbNotIncludedMP[i])
                continue;

            MapLine *pML = vpML[i];

            if (pML->isBad())
                continue;

            g2o::VertexSBAPointXYZ *vStartP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + 1));
            g2o::VertexSBAPointXYZ *vEndP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pML->mnId + maxKFid + maxMapPointID + vpML.size() + 1));

            if (nLoopKF == 0)
            {
                Vector6d LinePos;
                LinePos << Converter::toVector3d(Converter::toCvMat(vStartP->estimate())), Converter::toVector3d(Converter::toCvMat(vEndP->estimate()));
                pML->SetWorldPos(LinePos);
                pML->UpdateAverageDir();
                pML->UpdateManhAxis();
            }
            else
            {
                pML->mPosGBA.create(6, 1, CV_32F);
                Converter::toCvMat(vStartP->estimate()).copyTo(pML->mPosGBA.rowRange(0, 3));
                Converter::toCvMat(vEndP->estimate()).copyTo(pML->mPosGBA.rowRange(3, 6));
            }
        }
    }

    int Optimizer::PoseOptimization(Frame *pFrame)
    {
        double invSigma = 1;

        // Set the Solver
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        // optimizer.setVerbose(true);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);
        const float deltaStereo = sqrt(7.815);
        const float deltaLend = sqrt(3.84);
        const float deltaManh = sqrt(0.05);

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; i++)
            {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP)
                {
                    // Monocular observation
                    if (pFrame->mvuRight[i] < 0)
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        Eigen::Matrix<double, 2, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaMono);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesMono.push_back(e);
                        vnIndexEdgeMono.push_back(i);
                    }
                    else // Stereo observation
                    {
                        nInitialCorrespondences++;
                        pFrame->mvbOutlier[i] = false;

                        // SET EDGE
                        Eigen::Matrix<double, 3, 1> obs;
                        const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                        const float &kp_ur = pFrame->mvuRight[i];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                        e->setMeasurement(obs);
                        const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(deltaStereo);

                        e->fx = pFrame->fx;
                        e->fy = pFrame->fy;
                        e->cx = pFrame->cx;
                        e->cy = pFrame->cy;
                        e->bf = pFrame->mbf;
                        cv::Mat Xw = pMP->GetWorldPos();
                        e->Xw[0] = Xw.at<float>(0);
                        e->Xw[1] = Xw.at<float>(1);
                        e->Xw[2] = Xw.at<float>(2);

                        optimizer.addEdge(e);

                        vpEdgesStereo.push_back(e);
                        vnIndexEdgeStereo.push_back(i);
                    }
                }
            }
        }

        // Set MapLine vertices
        const int NL = pFrame->NL;
        int nLineInitalCorrespondences = 0;

        // Start-Point
        vector<DistPt2Line2DMultiFrameOnlyPose *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);

        // End-Point
        vector<DistPt2Line2DMultiFrameOnlyPose *> vpEdgesLineEp;
        vector<size_t> vnIndexLineEdgeEp;
        vpEdgesLineEp.reserve(NL);
        vnIndexLineEdgeEp.reserve(NL);

        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++)
            {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML)
                {
                    nLineInitalCorrespondences++;
                    pFrame->mvbLineOutlier[i] = false;

                    // 2D line equation
                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    // Start Point Edge
                    DistPt2Line2DMultiFrameOnlyPose *els = new DistPt2Line2DMultiFrameOnlyPose();

                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(deltaLend);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    els->Xw = pML->mWorldPos.head(3);

                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    // End Point Edge
                    DistPt2Line2DMultiFrameOnlyPose *ele = new DistPt2Line2DMultiFrameOnlyPose();

                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(deltaLend);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    ele->Xw = pML->mWorldPos.tail(3);

                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                    vnIndexLineEdgeEp.push_back(i);
                }
            }
        }

        if (nInitialCorrespondences < 3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const float chi2Stereo[4] = {7.815, 7.815, 7.815, 7.815};
        const float chi2LEnd[4] = {3.84, 3.84, 3.84, 3.84};
        const float chi2Manh[4] = {0.05, 0.05, 0.025, 0.01};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0;     
        int nLineBad = 0; 
        for (size_t it = 0; it < 4; it++)
        {
            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);
            nBad = 0;

            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Mono[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZOnlyPose *e = vpEdgesStereo[i];

                const size_t idx = vnIndexEdgeStereo[i];

                if (pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Stereo[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    e->setLevel(0);
                    pFrame->mvbOutlier[idx] = false;
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            nLineBad = 0;
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++)
            {
                DistPt2Line2DMultiFrameOnlyPose *e1 = vpEdgesLineSp[i];
                DistPt2Line2DMultiFrameOnlyPose *e2 = vpEdgesLineEp[i];

                const size_t idx = vnIndexLineEdgeSp[i];

                if (pFrame->mvbLineOutlier[idx])
                {
                    e1->computeError();
                    e2->computeError();
                }

                const float chi2_s = e1->chi2();
                const float chi2_e = e2->chi2();

                if (chi2_s > chi2LEnd[it] && chi2_e > chi2LEnd[it])
                {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nLineBad++;
                }
                else
                {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                }

                if (it == 2)
                {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }
            int inliers = 0;
            int total = 0;

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad - nLineBad;
    }

    int Optimizer::LineOptStruct(Frame *pFrame)
    {
        double invSigma = 1.0;

        // Set the Solver
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::LandmarkMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        const int NL = pFrame->NL;
        const float thHuberLine = sqrt(0.02);

        // Reserve size of the vectors required
        int par_obs_size = 0;
        for (size_t i = 0; i < pFrame->mvParLinesIdx.size(); i++)
        {
            par_obs_size += pFrame->mvParLinesIdx[i]->size();
        }

        int perp_obs_size = 0;
        for (size_t i = 0; i < pFrame->mvParLinesIdx.size(); i++)
        {
            perp_obs_size += pFrame->mvPerpLinesIdx[i]->size();
        }

        // vector<ParEptsNVector2DSingleFrame*> vpEdgesLineVecPar;
        vector<ParEptsNVector3DSingleFrame *> vpEdgesLineVecPar;

        vector<size_t> vnIndexLineEdgePar;
        vector<size_t> vnIndexLineEdgeParObs;

        vpEdgesLineVecPar.reserve(par_obs_size);
        vnIndexLineEdgePar.reserve(par_obs_size);
        vnIndexLineEdgeParObs.reserve(par_obs_size);

        vector<PerpEptsNVector3DSingleFrame *> vpEdgesLineVecPerp;
        // vector<PerpEptsNVector2DSingleFrame*> vpEdgesLineVecPerp;
        vector<size_t> vnIndexLineEdgePerpObs;
        vector<size_t> vnIndexLineEdgePerp;

        vpEdgesLineVecPerp.reserve(perp_obs_size);
        vnIndexLineEdgePerp.reserve(perp_obs_size);
        vnIndexLineEdgePerpObs.reserve(perp_obs_size);

        std::vector<bool> b_par_obs(par_obs_size, false);
        std::vector<bool> b_perp_obs(perp_obs_size, false);

        int idx_par = 0;
        int idx_perp = 0;
        int n_lines_to_opt = 0;

        // Add edges and vertex to the optimization problem
        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);
            for (int i = 0; i < NL; i++)
            {
                if (pFrame->mvParLinesIdx[i]->size() + pFrame->mvPerpLinesIdx[i]->size() < 5)
                {
                    continue;
                }

                std::pair<Eigen::Vector3d, Eigen::Vector3d> line_epts = pFrame->mvLines3D[i];
                Eigen::Vector3d st_pt = line_epts.first;
                Eigen::Vector3d end_pt = line_epts.second;

                if (st_pt[2] == 0.0 || st_pt[0] == -1.0)
                    continue;
                if (end_pt[2] == 0.0 || end_pt[0] == -1.0)
                    continue;

                if (abs(end_pt[0] - st_pt[0]) < 0.00001 && abs(end_pt[1] - st_pt[1]) < 0.00001)
                    continue;

                if (isnan(st_pt[0]) || isnan(st_pt[1]) || isnan(st_pt[2]))
                    continue;
                if (isnan(end_pt[0]) || isnan(end_pt[1]) || isnan(end_pt[2]))
                    continue;

                n_lines_to_opt++;

                g2o::VertexSBAPointXYZ *vStPt = new g2o::VertexSBAPointXYZ();
                vStPt->setEstimate(st_pt);
                int id_st = 2 * i;
                vStPt->setId(id_st);
                optimizer.addVertex(vStPt);

                g2o::VertexSBAPointXYZ *vEndPt = new g2o::VertexSBAPointXYZ();
                vEndPt->setEstimate(end_pt);
                int id_end = 2 * i + 1;
                vEndPt->setId(id_end);
                optimizer.addVertex(vEndPt);

                for (size_t j = 0; j < pFrame->mvParLinesIdx[i]->size(); j++)
                {
                    if (pFrame->mvParLinesIdx[i]->at(j) < 0)
                        continue;

                    cv::Vec3f line_obs_3d_t = pFrame->mvLineEq[pFrame->mvParLinesIdx[i]->at(j)];

                    if (line_obs_3d_t[2] == 0.0 || line_obs_3d_t[0] == -1.0)
                        continue;

                    b_par_obs[idx_par] = false;

                    Eigen::Vector3d line_obs;
                    line_obs << double(line_obs_3d_t[0]), double(line_obs_3d_t[1]), double(line_obs_3d_t[2]);

                    // If Optimize the structural constraints using the 2D line obs.
                    // Eigen::Vector3d line_obs;
                    // line_obs = pFrame->mvKeyLineFunctions[pFrame->mvParLinesIdx[i]->at(j)];
                    // if (line_obs[2] == 0.0 || line_obs[0] == -1.0)
                    //     continue;
                    // g2o::ParEptsNVector2DSingleFrame *l_vec = new g2o::ParEptsNVector2DSingleFrame();

                    // l_vec->fx = pFrame->fx;
                    // l_vec->fy = pFrame->fy;
                    // l_vec->cx = pFrame->cx;
                    // l_vec->cy = pFrame->cy;

                    // If optimize the structural constraints using the 3D line obs.
                    float response = pFrame->mvKeylinesUn[pFrame->mvParLinesIdx[i]->at(j)].response;
                    g2o::ParEptsNVector3DSingleFrame *l_vec = new g2o::ParEptsNVector3DSingleFrame();

                    l_vec->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id_st)));
                    l_vec->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id_end)));
                    l_vec->setMeasurement(line_obs);
                    l_vec->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    l_vec->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(thHuberLine);

                    optimizer.addEdge(l_vec);

                    vpEdgesLineVecPar.push_back(l_vec);
                    vnIndexLineEdgePar.push_back(i);
                    vnIndexLineEdgeParObs.push_back(j);

                    idx_par++;
                }

                for (size_t j = 0; j < pFrame->mvPerpLinesIdx[i]->size(); j++)
                {
                    if (pFrame->mvPerpLinesIdx[i]->at(j) < 0)
                        continue;
                    cv::Vec3f line_obs_3d_t = pFrame->mvLineEq[pFrame->mvPerpLinesIdx[i]->at(j)];

                    if (line_obs_3d_t[2] == 0.0 || line_obs_3d_t[0] == -1.0)
                        continue;
                    b_perp_obs[idx_perp] = false;

                    Eigen::Vector3d line_obs;
                    line_obs << double(line_obs_3d_t[0]), double(line_obs_3d_t[1]), double(line_obs_3d_t[2]);

                    // Eigen::Vector3d line_obs;
                    // line_obs = pFrame->mvKeyLineFunctions[pFrame->mvPerpLinesIdx[i]->at(j)];
                    // if (line_obs[2] == 0.0 || line_obs[0] == -1.0)
                    //     continue;
                    // b_perp_obs[idx_perp] = false;

                    float response = pFrame->mvKeylinesUn[pFrame->mvPerpLinesIdx[i]->at(j)].response;

                    g2o::PerpEptsNVector3DSingleFrame *l_vec = new g2o::PerpEptsNVector3DSingleFrame();
                    // g2o::PerpEptsNVector2DSingleFrame *l_vec = new g2o::PerpEptsNVector2DSingleFrame();

                    l_vec->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id_st)));
                    l_vec->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id_end)));
                    l_vec->setMeasurement(line_obs);
                    l_vec->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    // l_vec->fx = pFrame->fx;
                    // l_vec->fy = pFrame->fy;
                    // l_vec->cx = pFrame->cx;
                    // l_vec->cy = pFrame->cy;

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    l_vec->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(thHuberLine);

                    optimizer.addEdge(l_vec);

                    vpEdgesLineVecPerp.push_back(l_vec);
                    vnIndexLineEdgePerp.push_back(i);
                    vnIndexLineEdgePerpObs.push_back(j);

                    idx_perp++;
                }
            }
        }

        // We perform 2 optimizations, after each optimization we classify observation as inlier/outlier
        const float chi2Manh[4] = {0.02, 0.01, 0.01, 0.01};

        int nBad = 0;    
        int nLineBad = 0;

        std::vector<int> inliers_per_line;

        for (size_t it = 0; it < 2; it++)
        {
            optimizer.initializeOptimization(0);
            optimizer.optimize(5);

            nBad = 0;

            std::vector<int> inliers_per_line_it(NL, 0);

            for (size_t i = 0, iend = vpEdgesLineVecPar.size(); i < iend; i++)
            {
                // ParEptsNVector2DSingleFrame *e1 = vpEdgesLineVecPar[i];
                ParEptsNVector3DSingleFrame *e1 = vpEdgesLineVecPar[i];

                if (b_par_obs[i])
                {
                    e1->computeError();
                }

                const float chi2_s = e1->chi2();

                if (chi2_s > chi2Manh[it])
                {
                    b_par_obs[i] = true;
                    e1->setLevel(1);
                }
                else
                {
                    b_par_obs[i] = false;
                    inliers_per_line_it[vnIndexLineEdgeParObs[i]]++;

                    e1->setLevel(0);
                }

                if (it == 2)
                {
                    e1->setRobustKernel(0);
                }
            }

            for (size_t i = 0, iend = vpEdgesLineVecPerp.size(); i < iend; i++)
            {
                PerpEptsNVector3DSingleFrame *e1 = vpEdgesLineVecPerp[i];
                // PerpEptsNVector2DSingleFrame *e1 = vpEdgesLineVecPerp[i];

                if (b_perp_obs[i])
                {
                    e1->computeError();
                }

                const float chi2_s = e1->chi2();

                if (chi2_s > chi2Manh[it])
                {
                    b_perp_obs[i] = true;
                    e1->setLevel(1);
                }
                else
                {
                    b_perp_obs[i] = false;
                    e1->setLevel(0);
                    inliers_per_line_it[vnIndexLineEdgePerpObs[i]]++;
                }

                if (it == 2)
                {
                    e1->setRobustKernel(0);
                }
            }

            if (optimizer.edges().size() < 10)
                break;

            inliers_per_line = inliers_per_line_it;
        }

        // Reject bad parallel observations
        for (size_t i = 0, iend = vpEdgesLineVecPar.size(); i < iend; i++)
        {
            // g2o::ParEptsNVector2DSingleFrame *e1 = vpEdgesLineVecPar[i];
            g2o::ParEptsNVector3DSingleFrame *e1 = vpEdgesLineVecPar[i];

            if (!(e1->chi2() >= 0.0 && e1->chi2() <= 0.02))
                pFrame->mvParLinesIdx[vnIndexLineEdgePar[i]]->at(vnIndexLineEdgeParObs[i]) = -1;
        }

        // Reject bad perpendicular observations
        for (size_t i = 0, iend = vpEdgesLineVecPerp.size(); i < iend; i++)
        {
            g2o::PerpEptsNVector3DSingleFrame *e1 = vpEdgesLineVecPerp[i];
            // g2o::PerpEptsNVector2DSingleFrame *e1 = vpEdgesLineVecPerp[i];

            if (!(e1->chi2() >= 0.0 && e1->chi2() <= 0.02))
                pFrame->mvPerpLinesIdx[vnIndexLineEdgePerp[i]]->at(vnIndexLineEdgePerpObs[i]) = -1;
        }

        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            int n_lines_opt = 0;

            for (int i = 0; i < NL; i++)
            {
                // Evaluates if this vertex IDX exists in the optimization process
                if (optimizer.vertex(2 * i) == 0 || optimizer.vertex((2 * i + 1) == 0))
                    continue;

                g2o::VertexSBAPointXYZ *vStPt = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(2 * i));
                g2o::VertexSBAPointXYZ *vEndPt = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(2 * i + 1));

                Eigen::Vector3d opt_st_pt = vStPt->estimate();
                Eigen::Vector3d opt_end_pt = vEndPt->estimate();
                // Evaluates if the resulting endpts values are correct
               
                std::pair<Eigen::Vector3d, Eigen::Vector3d> opt_line_epts(opt_st_pt, opt_end_pt);
                pFrame->mvLines3D[i] = opt_line_epts;
                n_lines_opt++;
            }
        }
    }

    int Optimizer::PoseOptimizationWithPoints(Frame *pFrame)
    {
        double invSigma = 1;
        
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>(); 

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        int nInitialCorrespondences = 0;

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        // Set MapPoint vertices
        const int N = pFrame->N;

        vector<g2o::EdgeSE3ProjectXYZOnlyPose *> vpEdgesMono;
        vector<size_t> vnIndexEdgeMono;
        vpEdgesMono.reserve(N);
        vnIndexEdgeMono.reserve(N);

        vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose *> vpEdgesStereo;
        vector<size_t> vnIndexEdgeStereo;
        vpEdgesStereo.reserve(N);
        vnIndexEdgeStereo.reserve(N);

        const float deltaMono = sqrt(5.991);

        {
            unique_lock<mutex> lock(MapPoint::mGlobalMutex);

            for (int i = 0; i < N; i++)
            {
                MapPoint *pMP = pFrame->mvpMapPoints[i];
                if (pMP)
                {
                    nInitialCorrespondences++;
                    pFrame->mvbOutlier[i] = false;

                    Eigen::Matrix<double, 2, 1> obs;
                    const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZOnlyPose *e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    e->setMeasurement(obs);
                    const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(deltaMono);

                    e->fx = pFrame->fx;
                    e->fy = pFrame->fy;
                    e->cx = pFrame->cx;
                    e->cy = pFrame->cy;
                    cv::Mat Xw = pMP->GetWorldPos();
                    e->Xw[0] = Xw.at<float>(0);
                    e->Xw[1] = Xw.at<float>(1);
                    e->Xw[2] = Xw.at<float>(2);

                    optimizer.addEdge(e);

                    vpEdgesMono.push_back(e);
                    vnIndexEdgeMono.push_back(i);
                }
            }
        }

        if (nInitialCorrespondences < 3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2Mono[4] = {5.991, 5.991, 5.991, 5.991};
        const int its[4] = {10, 10, 10, 10};

        int nBad = 0; 
        for (size_t it = 0; it < 4; it++)
        {

            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nBad = 0;
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose *e = vpEdgesMono[i];

                const size_t idx = vnIndexEdgeMono[i];

                if (pFrame->mvbOutlier[idx])
                {
                    e->computeError();
                }

                const float chi2 = e->chi2();

                if (chi2 > chi2Mono[it])
                {
                    pFrame->mvbOutlier[idx] = true;
                    e->setLevel(1);
                    nBad++;
                }
                else
                {
                    pFrame->mvbOutlier[idx] = false;
                    e->setLevel(0);
                }

                if (it == 2)
                    e->setRobustKernel(0);
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nInitialCorrespondences - nBad;
    }

    int Optimizer::PoseOptimizationWithLines(Frame *pFrame)
    {
        double invSigma = 1;
        
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>(); 

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Set Frame vertex
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        vSE3->setId(0);
        vSE3->setFixed(false);
        optimizer.addVertex(vSE3);

        const float deltaLEnd = sqrt(3.84);

        // Set MapLine vertices
        const int NL = pFrame->NL;
        int nLineInitalCorrespondences = 0;

        vector<DistPt2Line2DMultiFrameOnlyPose *> vpEdgesLineSp;
        vector<size_t> vnIndexLineEdgeSp;
        vpEdgesLineSp.reserve(NL);

        vector<DistPt2Line2DMultiFrameOnlyPose *> vpEdgesLineEp;
        vector<size_t> vnIndexLineEdgeEp;
        vpEdgesLineEp.reserve(NL);
        vnIndexLineEdgeEp.reserve(NL);

        {
            unique_lock<mutex> lock(MapLine::mGlobalMutex);

            for (int i = 0; i < NL; i++)
            {
                MapLine *pML = pFrame->mvpMapLines[i];
                if (pML)
                {
                    nLineInitalCorrespondences++;
                    pFrame->mvbLineOutlier[i] = false;

                    Eigen::Vector3d line_obs;
                    line_obs = pFrame->mvKeyLineFunctions[i];

                    DistPt2Line2DMultiFrameOnlyPose *els = new DistPt2Line2DMultiFrameOnlyPose();

                    els->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    els->setMeasurement(line_obs);
                    els->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    els->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(deltaLEnd);

                    els->fx = pFrame->fx;
                    els->fy = pFrame->fy;
                    els->cx = pFrame->cx;
                    els->cy = pFrame->cy;

                    els->Xw = pML->mWorldPos.head(3);

                    optimizer.addEdge(els);

                    vpEdgesLineSp.push_back(els);
                    vnIndexLineEdgeSp.push_back(i);

                    // 特征点的终止点
                    DistPt2Line2DMultiFrameOnlyPose *ele = new DistPt2Line2DMultiFrameOnlyPose();

                    ele->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
                    ele->setMeasurement(line_obs);
                    ele->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    ele->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(deltaLEnd);

                    ele->fx = pFrame->fx;
                    ele->fy = pFrame->fy;
                    ele->cx = pFrame->cx;
                    ele->cy = pFrame->cy;

                    ele->Xw = pML->mWorldPos.tail(3);

                    optimizer.addEdge(ele);

                    vpEdgesLineEp.push_back(ele);
                    vnIndexLineEdgeEp.push_back(i);
                }
            }
        }

        if (nLineInitalCorrespondences < 3)
            return 0;

        // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
        // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
        const float chi2LEnd[4] = {3.84, 3.84, 3.84, 3.84};
        const int its[4] = {10, 10, 10, 10};

        int nLineBad = 0; //线特征
        for (size_t it = 0; it < 4; it++)
        {
            vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
            optimizer.initializeOptimization(0);
            optimizer.optimize(its[it]);

            nLineBad = 0;
            for (size_t i = 0, iend = vpEdgesLineSp.size(); i < iend; i++)
            {
                DistPt2Line2DMultiFrameOnlyPose *e1 = vpEdgesLineSp[i]; 
                DistPt2Line2DMultiFrameOnlyPose *e2 = vpEdgesLineEp[i]; 

                const size_t idx = vnIndexLineEdgeSp[i]; 

                if (pFrame->mvbLineOutlier[idx])
                {
                    e1->computeError();
                    e2->computeError();
                }

                const float chi2_s = e1->chi2();
                const float chi2_e = e2->chi2();

                if (chi2_s > chi2LEnd[it] || chi2_e > chi2LEnd[it])
                {
                    pFrame->mvbLineOutlier[idx] = true;
                    e1->setLevel(1);
                    e2->setLevel(1);
                    nLineBad++;
                }
                else
                {
                    pFrame->mvbLineOutlier[idx] = false;
                    e1->setLevel(0);
                    e2->setLevel(0);
                }

                if (it == 2)
                {
                    e1->setRobustKernel(0);
                    e2->setRobustKernel(0);
                }
            }

            if (optimizer.edges().size() < 10)
                break;
        }

        // Recover optimized pose and return number of inliers
        g2o::VertexSE3Expmap *vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        cv::Mat pose = Converter::toCvMat(SE3quat_recov);
        pFrame->SetPose(pose);

        return nLineInitalCorrespondences - nLineBad;
    }


    void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
    {
        // Local KeyFrames: First Breath Search from Current Keyframe
        list<KeyFrame *> lLocalKeyFrames;

        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;

        const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
        {
            KeyFrame *pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId; 
            if (!pKFi->isBad())
                lLocalKeyFrames.push_back(pKFi);
        }

        // Local MapPoints seen in Local KeyFrames
        list<MapPoint *> lLocalMapPoints;
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
            for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                MapPoint *pMP = *vit;
                if (pMP)
                    if (!pMP->isBad())
                        if (pMP->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
            }
        }

        // Fixed Keyframes. Keyframes that see Local MapPoints but that are not Local Keyframes
        list<KeyFrame *> lFixedCameras;
        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
            for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    if (!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }

        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();

        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        if (pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;

        // Set Local KeyFrame vertices
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // Set Fixed KeyFrame vertices
        for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // Set MapPoint vertices
        const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

        vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<KeyFrame *> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<MapPoint *> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        vector<g2o::EdgeStereoSE3ProjectXYZ *> vpEdgesStereo;
        vpEdgesStereo.reserve(nExpectedSize);

        vector<KeyFrame *> vpEdgeKFStereo;
        vpEdgeKFStereo.reserve(nExpectedSize);

        vector<MapPoint *> vpMapPointEdgeStereo;
        vpMapPointEdgeStereo.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberStereo = sqrt(7.815);
        const float thHuberLEnd = sqrt(3.84);

        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            // Set edges
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (!pKFi->isBad())
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                    // Monocular observation
                    if (pKFi->mvuRight[mit->second] < 0)
                    {
                        Eigen::Matrix<double, 2, 1> obs;
                        obs << kpUn.pt.x, kpUn.pt.y;

                        g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberMono);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;

                        optimizer.addEdge(e);
                        vpEdgesMono.push_back(e);
                        vpEdgeKFMono.push_back(pKFi);
                        vpMapPointEdgeMono.push_back(pMP);
                    }
                    else // Stereo observation
                    {
                        Eigen::Matrix<double, 3, 1> obs;
                        const float kp_ur = pKFi->mvuRight[mit->second];
                        obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                        g2o::EdgeStereoSE3ProjectXYZ *e = new g2o::EdgeStereoSE3ProjectXYZ();

                        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                        e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                        e->setMeasurement(obs);
                        const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                        Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                        e->setInformation(Info);

                        g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                        e->setRobustKernel(rk);
                        rk->setDelta(thHuberStereo);

                        e->fx = pKFi->fx;
                        e->fy = pKFi->fy;
                        e->cx = pKFi->cx;
                        e->cy = pKFi->cy;
                        e->bf = pKFi->mbf;

                        optimizer.addEdge(e);
                        vpEdgesStereo.push_back(e);
                        vpEdgeKFStereo.push_back(pKFi);
                        vpMapPointEdgeStereo.push_back(pMP);
                    }
                }
            }
        }

        if (pbStopFlag)
            if (*pbStopFlag)
                return;

        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool bDoMore = true;

        if (pbStopFlag)
            if (*pbStopFlag)
                bDoMore = false;

        if (bDoMore)
        {

            // Check inlier observations
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
                MapPoint *pMP = vpMapPointEdgeMono[i];

                if (pMP->isBad())
                    continue;

                if (e->chi2() > 5.991 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
            {
                g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
                MapPoint *pMP = vpMapPointEdgeStereo[i];

                if (pMP->isBad())
                    continue;

                if (e->chi2() > 7.815 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            // Optimize again without the outliers
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }

        vector<pair<KeyFrame *, MapPoint *>> vToErase;
        vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                KeyFrame *pKFi = vpEdgeKFMono[i];
                vToErase.push_back(make_pair(pKFi, pMP));
            }
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZ *e = vpEdgesStereo[i];
            MapPoint *pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 7.815 || !e->isDepthPositive())
            {
                KeyFrame *pKFi = vpEdgeKFStereo[i];
                vToErase.push_back(make_pair(pKFi, pMP));
            }
        }

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        if (!vToErase.empty())
        {
            for (size_t i = 0; i < vToErase.size(); i++)
            {
                KeyFrame *pKFi = vToErase[i].first;
                MapPoint *pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }

        // Recover optimized data

        // Keyframes
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            KeyFrame *pKF = *lit;
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }

        // Points
        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
    }

    void Optimizer::LocalBundleAdjustmentWithLine(const cv::Mat &manh_axis, KeyFrame *pKF, bool *pbStopFlag, Map *pMap, const bool &MAxisAval)
    {
        // Higher values more confidence. Maxis highe values
        double invSigma = 0.3;
        // Local KeyFrames: First Breath Search from Current KeyFrame
        list<KeyFrame *> lLocalKeyFrames;

        // step1: Add the current keyframe to lLocalKeyFrames
        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;

        // step2:Find the keyframe connected by the keyframe (first level connection), and add it to lLocalKeyFrames
        const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
        {
            KeyFrame *pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId;
            if (!pKFi->isBad())
                lLocalKeyFrames.push_back(pKFi);
        }

        list<MapPoint *> lLocalMapPoints;
        // step3：Add MapPoints of lLocalKeyFrames to lLocalMapPoints
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
            for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                MapPoint *pMP = *vit;
                if (pMP)
                {
                    if (!pMP->isBad())
                    {
                        if (pMP->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        list<MapLine *> lLocalMapLines;
        // step4: use lLocalKeyFrames to extract the MapLines that can be observed in each key frame, and put them in lLocalMapLines
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            vector<MapLine *> vpMLs = (*lit)->GetMapLineMatches();
            for (vector<MapLine *>::iterator vit = vpMLs.begin(), vend = vpMLs.end(); vit != vend; vit++)
            {
                MapLine *pML = *vit;
                if (pML)
                {
                    if (!pML->isBad())
                    {
                        if (pML->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapLines.push_back(pML);
                            pML->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        list<KeyFrame *> lFixedCameras;
        // step5: Keyframes that are observed by local MapPoints but are not local keyframes, these keyframes are fixed during local BA optimization

        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
            for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    if (!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }

        // step6:Keyframes that are observed by local MapLines, but do not belong to local keyframes. These keyframes are fixed during local BA optimization
        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++)
        {
            map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
            for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    if (!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }

        // step6：Constructor g2o optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

        //  g2o::BlockSolver_6_3::LinearSolverType* linearSolver;
        // linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        // g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        if (pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;
        // step7：Add vertices of the Pose of Local KeyFrames
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // step8:Add the vertices of the Pose Fixed KeyFrame
        for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        vector<int> MapPointID;
        //***********************Set MapPoint Vertices******************************
        // step9：Add 3D vertices of MapPoint
        const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

        vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<KeyFrame *> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<MapPoint *> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberLEnd = sqrt(3.84);
        const float thHuberLPar = sqrt(0.08);
        // const float thHuberLManh = sqrt(0.12);

        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            MapPointID.push_back(id);

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            // Set Edges
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (!pKFi->isBad())
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                    // Monocular observation
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
            }
        }

        std::sort(MapPointID.begin(), MapPointID.end());
        int maxMapPointID = MapPointID[MapPointID.size() - 1];

        //***********************Set MapLine Vertices******************************
        // step10：Add the vertices of the MapLine
        const int nLineExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapLines.size();

        vector<DistPt2Line2DMultiFrame *> vpLineEdgesSP;
        vpLineEdgesSP.reserve(nLineExpectedSize);

        vector<DistPt2Line2DMultiFrame *> vpLineEdgesEP;
        vpLineEdgesEP.reserve(nLineExpectedSize);

        vector<KeyFrame *> vpLineEdgeKF;
        vpLineEdgeKF.reserve(nLineExpectedSize);

        vector<MapLine *> vpMapLineEdge;
        vpMapLineEdge.reserve(nLineExpectedSize);

        vector<ParEptsNVector3DSingleFrame *> vpLineManh;
        vpLineManh.reserve(nLineExpectedSize);

        vector<KeyFrame *> vpLineManhKF;
        vpLineManhKF.reserve(nLineExpectedSize);

        vector<MapLine *> vpMapLineManhEdge;
        vpMapLineManhEdge.reserve(nLineExpectedSize);

        // Add line vertex
        vector<int> MapLineID;

        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++)
        {
            MapLine *pML = *lit;

            Vector6d world_pose = pML->GetWorldPos();

            g2o::VertexSBAPointXYZ *vStartP = new g2o::VertexSBAPointXYZ();
            vStartP->setEstimate(world_pose.head(3));
            int ids = 2 * pML->mnId + maxMapPointID + 1;
            vStartP->setId(ids);
            optimizer.addVertex(vStartP);

            g2o::VertexSBAPointXYZ *vEndP = new VertexSBAPointXYZ();
            vEndP->setEstimate(world_pose.tail(3));
            int ide = 2 * pML->mnId + maxMapPointID + 2;
            MapLineID.push_back(ide);

            vEndP->setId(ide);
            optimizer.addVertex(vEndP);

            const map<KeyFrame *, size_t> observations = pML->GetObservations();

            // Set Edges Line reprojection Error
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (!pKFi->isBad())
                {
                    Eigen::Vector3d line_obs;
                    line_obs = pKFi->mvKeyLineFunctions[mit->second];

                    // StartPoint
                    DistPt2Line2DMultiFrame *e1 = new DistPt2Line2DMultiFrame();

                    e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ids)));
                    e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e1->setMeasurement(line_obs);
                    e1->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    e1->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(thHuberLEnd);

                    e1->fx = pKFi->fx;
                    e1->fy = pKFi->fy;
                    e1->cx = pKFi->cx;
                    e1->cy = pKFi->cy;

                    optimizer.addEdge(e1);
                    vpLineEdgesSP.push_back(e1);
                    vpLineEdgeKF.push_back(pKFi);
                    vpMapLineEdge.push_back(pML);

                    // EndPoint
                    DistPt2Line2DMultiFrame *e2 = new DistPt2Line2DMultiFrame();

                    e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ide)));
                    e2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e2->setMeasurement(line_obs);
                    e2->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    e2->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(thHuberLEnd);

                    e2->fx = pKF->fx;
                    e2->fy = pKF->fy;
                    e2->cx = pKF->cx;
                    e2->cy = pKF->cy;

                    optimizer.addEdge(e2);
                    vpLineEdgesEP.push_back(e2);
                    vpLineEdgeKF.push_back(pKFi);
                }
            }
        }

        std::sort(MapLineID.begin(), MapLineID.end());
        int maxMapLineID = MapLineID[MapLineID.size() - 1];

        if (pbStopFlag)
            if (*pbStopFlag)
                return;

        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool bDoMore = true;

        if (pbStopFlag)
            if (*pbStopFlag)
                bDoMore = false;
        if (bDoMore)
        {
            // Check inlier observations
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
                MapPoint *pMP = vpMapPointEdgeMono[i];

                if (pMP->isBad())
                    continue;

                if (e->chi2() > 5.991 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }
                e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpMapLineEdge.size(); i < iend; i++)
            {
                DistPt2Line2DMultiFrame *e1 = vpLineEdgesSP[i];
                DistPt2Line2DMultiFrame *e2 = vpLineEdgesEP[i];
                MapLine *pML = vpMapLineEdge[i];

                if (pML->isBad())
                    continue;

                if (e1->chi2() > 3.84 || e2->chi2() > 3.84)
                {
                    e1->setLevel(1);
                    e2->setLevel(1);
                }
                e1->setRobustKernel(0);
                e2->setRobustKernel(0);
            }
            // Optimize again without the outliers
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }

        vector<pair<KeyFrame *, MapPoint *>> vToErase;
        vToErase.reserve(vpEdgesMono.size());
        // check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                KeyFrame *pKFi = vpEdgeKFMono[i];
                vToErase.push_back(make_pair(pKFi, pMP));
            }
        }

        vector<pair<KeyFrame *, MapLine *>> vLineToErase;
        vLineToErase.reserve(2 * vpLineEdgesSP.size());
        for (size_t i = 0, iend = vpLineEdgesSP.size(); i < iend; i++)
        {
            DistPt2Line2DMultiFrame *e1 = vpLineEdgesSP[i];
            DistPt2Line2DMultiFrame *e2 = vpLineEdgesSP[i];
            MapLine *pML = vpMapLineEdge[i];

            if (pML->isBad())
                continue;

            if (e1->chi2() > 3.84 || e2->chi2() > 3.84)
            {
                KeyFrame *pKFi = vpLineEdgeKF[i];
                vLineToErase.push_back(make_pair(pKFi, pML));
            }
        }

        int counter_out = 0;
        vector<pair<KeyFrame *, MapLine *>> vLineObsManhToErase;
        vLineObsManhToErase.reserve(vpMapLineManhEdge.size());
        for (size_t i = 0, iend = vpLineManh.size(); i < iend; i++)
        {
            ParEptsNVector3DSingleFrame *e1 = vpLineManh[i];
            MapLine *pML = vpMapLineManhEdge[i];

            if (pML->isBad())
                continue;

            if (e1->chi2() > 0.13)
            {
                KeyFrame *pKFi = vpLineManhKF[i];
                vLineObsManhToErase.push_back(make_pair(pKFi, pML));
                counter_out++;
            }
        }

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        if (!vToErase.empty())
        {
            for (size_t i = 0; i < vToErase.size(); i++)
            {
                KeyFrame *pKFi = vToErase[i].first;
                MapPoint *pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        if (!vLineToErase.empty())
        {
            for (size_t i = 0; i < vLineToErase.size(); i++)
            {
                KeyFrame *pKFi = vLineToErase[i].first;
                MapLine *pMLi = vLineToErase[i].second;
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseObservation(pKFi);
            }
        }
        if (!vLineObsManhToErase.empty())
        {
            for (size_t i = 0; i < vLineObsManhToErase.size(); i++)
            {
                KeyFrame *pKFi = vLineObsManhToErase[i].first;
                MapLine *pML = vLineObsManhToErase[i].second;
                pML->EraseManhObs(pKFi);
            }
        }

        // Recover optimized data
        // Keyframes
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            KeyFrame *pKF = *lit;
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        // Points
        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        // Lines
        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++)
        {
            MapLine *pML = *lit;

            // Check if the vertex exist
            if (optimizer.vertex(2 * pML->mnId + maxMapPointID + 1) == 0)
            {
                continue;
            }

            g2o::VertexSBAPointXYZ *vStartP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(2 * pML->mnId + maxMapPointID + 1));
            g2o::VertexSBAPointXYZ *vEndP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(2 * pML->mnId + maxMapPointID + 2));

            Vector6d LinePos;
            LinePos << Converter::toVector3d(Converter::toCvMat(vStartP->estimate())), Converter::toVector3d(Converter::toCvMat(vEndP->estimate()));
            pML->SetWorldPos(LinePos);
            pML->UpdateAverageDir();
            pML->UpdateManhAxis();
        }
    }

    void Optimizer::LocalMapOptimization(const cv::Mat &manh_axis, KeyFrame *pKF, bool *pbStopFlag, Map *pMap, const bool &MAxisAval)
    {
        // Higher values more confidence. Maxis highe values
        double invSigma = 0.3;
        double invSigmaMAxis = 0.3;
        double invSigmaStruct = 0.5;
        // Local KeyFrames: First Breath Search from Current KeyFrame
        list<KeyFrame *> lLocalKeyFrames;

        // step1: Add the current keyframe to lLocalKeyFrames
        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;

        // step2:Find the keyframe connected by the keyframe (first level connection), and add it to lLocalKeyFrames
        const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
        {
            KeyFrame *pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId;
            if (!pKFi->isBad())
                lLocalKeyFrames.push_back(pKFi);
        }

        list<MapPoint *> lLocalMapPoints;
        // step3：Add MapPoints of lLocalKeyFrames to lLocalMapPoints
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            vector<MapPoint *> vpMPs = (*lit)->GetMapPointMatches();
            for (vector<MapPoint *>::iterator vit = vpMPs.begin(), vend = vpMPs.end(); vit != vend; vit++)
            {
                MapPoint *pMP = *vit;
                if (pMP)
                {
                    if (!pMP->isBad())
                    {
                        if (pMP->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapPoints.push_back(pMP);
                            pMP->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        list<MapLine *> lLocalMapLines;
        // step4: use lLocalKeyFrames to extract the MapLines that can be observed in each key frame, and put them in lLocalMapLines
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            vector<MapLine *> vpMLs = (*lit)->GetMapLineMatches();
            for (vector<MapLine *>::iterator vit = vpMLs.begin(), vend = vpMLs.end(); vit != vend; vit++)
            {
                MapLine *pML = *vit;
                if (pML)
                {
                    if (!pML->isBad())
                    {
                        if (pML->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapLines.push_back(pML);
                            pML->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        list<KeyFrame *> lFixedCameras;
        // step5: Keyframes that are observed by local MapPoints but are not local keyframes, these keyframes are fixed during local BA optimization

        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
            for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    if (!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }

        // step6:Keyframes that are observed by local MapLines, but do not belong to local keyframes. These keyframes are fixed during local BA optimization
        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++)
        {
            map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
            for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    if (!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }

        // step6：Constructor g2o optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        if (pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;
        // step7：Add vertices of the Pose of Local KeyFrames
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // step8:Add the vertices of the Pose Fixed KeyFrame
        for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        vector<int> MapPointID;
        //***********************Set MapPoint Vertices******************************
        // step9：Add 3D vertices of MapPoint
        const int nExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapPoints.size();

        vector<g2o::EdgeSE3ProjectXYZ *> vpEdgesMono;
        vpEdgesMono.reserve(nExpectedSize);

        vector<KeyFrame *> vpEdgeKFMono;
        vpEdgeKFMono.reserve(nExpectedSize);

        vector<MapPoint *> vpMapPointEdgeMono;
        vpMapPointEdgeMono.reserve(nExpectedSize);

        const float thHuberMono = sqrt(5.991);
        const float thHuberLEnd = sqrt(3.84);
        const float thHuberLPar = sqrt(0.08);

        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
            int id = pMP->mnId + maxKFid + 1;
            vPoint->setId(id);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
            MapPointID.push_back(id);

            const map<KeyFrame *, size_t> observations = pMP->GetObservations();

            // Set Edges
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (!pKFi->isBad())
                {
                    const cv::KeyPoint &kpUn = pKFi->mvKeysUn[mit->second];

                    // Monocular observation
                    Eigen::Matrix<double, 2, 1> obs;
                    obs << kpUn.pt.x, kpUn.pt.y;

                    g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();

                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
                    e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e->setMeasurement(obs);
                    const float &invSigma2 = pKFi->mvInvLevelSigma2[kpUn.octave];
                    e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                    g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuberMono);

                    e->fx = pKFi->fx;
                    e->fy = pKFi->fy;
                    e->cx = pKFi->cx;
                    e->cy = pKFi->cy;

                    optimizer.addEdge(e);
                    vpEdgesMono.push_back(e);
                    vpEdgeKFMono.push_back(pKFi);
                    vpMapPointEdgeMono.push_back(pMP);
                }
            }
        }

        std::sort(MapPointID.begin(), MapPointID.end());
        int maxMapPointID = MapPointID[MapPointID.size() - 1];

        //***********************Set MapLine Vertices******************************
        // step10：Add the vertices of the MapLine
        const int nLineExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapLines.size();

        vector<DistPt2Line2DMultiFrame *> vpLineEdgesSP;
        vpLineEdgesSP.reserve(nLineExpectedSize);

        vector<DistPt2Line2DMultiFrame *> vpLineEdgesEP;
        vpLineEdgesEP.reserve(nLineExpectedSize);

        vector<KeyFrame *> vpLineEdgeKF;
        vpLineEdgeKF.reserve(nLineExpectedSize);

        vector<MapLine *> vpMapLineEdge;
        vpMapLineEdge.reserve(nLineExpectedSize);

        vector<ParEptsNVector3DSingleFrame *> vpLineManh;
        vpLineManh.reserve(nLineExpectedSize);

        vector<KeyFrame *> vpLineManhKF;
        vpLineManhKF.reserve(nLineExpectedSize);

        vector<MapLine *> vpMapLineManhEdge;
        vpMapLineManhEdge.reserve(nLineExpectedSize);

        int size_par_obs = 0;
        int size_perp_obs = 0;

        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++)
        {
            MapLine *pML = *lit;
            size_par_obs += pML->GetParObservations().size();
            size_perp_obs += pML->GetPerpObservations().size();
        }

        vector<ParEptsNVector2DMultiFrame *> vpLineEdgesParObs;
        vpLineEdgesParObs.reserve(size_par_obs);

        vector<PerpEptsNVector2DMultiFrame *> vpLineEdgesPerpObs;
        vpLineEdgesPerpObs.reserve(size_perp_obs);

        vector<MapLine *> vpMapLineEdgePar;
        vpMapLineEdgePar.reserve(size_par_obs);

        vector<MapLine *> vpMapLineEdgePerp;
        vpMapLineEdgePerp.reserve(size_perp_obs);

        vector<KeyFrame *> vpLineEdgeKFPar;
        vpLineEdgeKFPar.reserve(size_par_obs);

        vector<KeyFrame *> vpLineEdgeKFPerp;
        vpLineEdgeKFPerp.reserve(size_perp_obs);

        vector<int> vpLineObsIdxPar;
        vpLineObsIdxPar.reserve(size_par_obs);

        vector<int> vpLineObsIdxPerp;
        vpLineObsIdxPerp.reserve(size_perp_obs);
        // Add line vertex
        vector<int> MapLineID;

        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++)
        {
            MapLine *pML = *lit;

            Vector6d world_pose = pML->GetWorldPos();

            if (isnan(world_pose[0]) || isnan(world_pose[1]) || isnan(world_pose[2]))
                continue;

            if (isnan(world_pose[3]) || isnan(world_pose[4]) || isnan(world_pose[5]))
                continue;

            if (abs(world_pose[3] - world_pose[0]) < 0.00001)
                continue;

            g2o::VertexSBAPointXYZ *vStartP = new g2o::VertexSBAPointXYZ();
            vStartP->setEstimate(world_pose.head(3));
            int ids = 2 * pML->mnId + maxMapPointID + 1;
            vStartP->setId(ids);
            optimizer.addVertex(vStartP);

            g2o::VertexSBAPointXYZ *vEndP = new VertexSBAPointXYZ();
            vEndP->setEstimate(world_pose.tail(3));
            int ide = 2 * pML->mnId + maxMapPointID + 2;
            MapLineID.push_back(ide);

            vEndP->setId(ide);
            optimizer.addVertex(vEndP);

            // Manhattan Axis Error
            // TODO1 : Add this variable to the YAML file
            bool use_MA = true;
            if (!manh_axis.empty() && MAxisAval && use_MA)
            {
                int manh_idx = pML->GetManhIdx();
                if (manh_idx > 0)
                {
                    cv::Mat eval_axis = manh_axis.col(manh_idx - 1);
                    Eigen::Vector3d manh_axis_obs(eval_axis.at<double>(0), eval_axis.at<double>(1), eval_axis.at<double>(2));

                    ParEptsNVector3DSingleFrame *e1 = new ParEptsNVector3DSingleFrame();

                    e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ids)));
                    e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ide)));
                    e1->setMeasurement(manh_axis_obs);
                    e1->setInformation(Eigen::Matrix3d::Identity() * invSigmaMAxis);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    e1->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(thHuberLPar);

                    optimizer.addEdge(e1);

                    vpLineManh.push_back(e1);
                    vpMapLineManhEdge.push_back(pML);
                }
            }

            const map<KeyFrame *, size_t> observations = pML->GetObservations();

            // Set Edges Line reprojection Error
            for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (!pKFi->isBad())
                {
                    Eigen::Vector3d line_obs;
                    line_obs = pKFi->mvKeyLineFunctions[mit->second];

                    // StartPoint
                    DistPt2Line2DMultiFrame *e1 = new DistPt2Line2DMultiFrame();

                    e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ids)));
                    e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e1->setMeasurement(line_obs);
                    e1->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                    e1->setRobustKernel(rk_line_s);
                    rk_line_s->setDelta(thHuberLEnd);

                    e1->fx = pKFi->fx;
                    e1->fy = pKFi->fy;
                    e1->cx = pKFi->cx;
                    e1->cy = pKFi->cy;

                    optimizer.addEdge(e1);
                    vpLineEdgesSP.push_back(e1);
                    vpLineEdgeKF.push_back(pKFi);
                    vpMapLineEdge.push_back(pML);

                    // EndPoint
                    DistPt2Line2DMultiFrame *e2 = new DistPt2Line2DMultiFrame();

                    e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ide)));
                    e2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    e2->setMeasurement(line_obs);
                    e2->setInformation(Eigen::Matrix3d::Identity() * invSigma);

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    e2->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(thHuberLEnd);

                    e2->fx = pKF->fx;
                    e2->fy = pKF->fy;
                    e2->cx = pKF->cx;
                    e2->cy = pKF->cy;

                    optimizer.addEdge(e2);
                    vpLineEdgesEP.push_back(e2);
                    vpLineEdgeKF.push_back(pKFi);
                }
            }
            // Line Structural Constraints
            std::map<KeyFrame *, std::vector<int>> ParObservations = pML->GetParObservations();
            int par_obs_size = pML->GetParObservations().size();
            for (const auto &observation : ParObservations)
            {
                KeyFrame *pKFi = observation.first;
                if (pKFi->isBad() || pKFi->mnId > maxKFid)
                    continue;

                for (size_t k = 0; k < observation.second.size(); k++)
                {
                    if (observation.second[k] < 0)
                        continue;
                    // 3D optimization
                    // cv::Vec3f kl = pKF->mvLineEq[observation.second[k]];
                    // if (kl[0] == 0.0 || kl[0] == -1.0)
                    //     continue;
                    // Eigen::Vector3d line_obs(kl[0], kl[1], kl[2]);

                    // 2D
                    Eigen::Vector3d line_obs;
                    line_obs = pKFi->mvKeyLineFunctions[observation.second[k]];
                    if (line_obs[0] == 0.0 || line_obs[0] == -1.0)
                        continue;

                    if (isnan(line_obs[0]) || isnan(line_obs[1]) || isnan(line_obs[2]))
                    {
                        continue;
                    }

                    // ParEptsNVector3DMultiFrame *pae = new ParEptsNVector3DMultiFrame();

                    ParEptsNVector2DMultiFrame *pae = new ParEptsNVector2DMultiFrame();

                    pae->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ids)));
                    pae->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ide)));
                    pae->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));

                    pae->setMeasurement(line_obs);
                    pae->setInformation(Eigen::Matrix3d::Identity() * (invSigmaStruct + (par_obs_size / 10.0)));

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    pae->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(thHuberLPar);

                    pae->fx = pKF->fx;
                    pae->fy = pKF->fy;
                    pae->cx = pKF->cx;
                    pae->cy = pKF->cy;

                    optimizer.addEdge(pae);

                    vpLineEdgesParObs.push_back(pae);
                    vpMapLineEdgePar.push_back(pML);
                    vpLineEdgeKFPar.push_back(pKFi);
                    vpLineObsIdxPar.push_back(k);
                }
            }

            int perp_obs_size = pML->GetPerpObservations().size();
            std::map<KeyFrame *, std::vector<int>> PerpObservations = pML->GetPerpObservations();
            for (const auto &observation : PerpObservations)
            {
                KeyFrame *pKFi = observation.first;
                if (pKFi->isBad() || pKFi->mnId > maxKFid)
                    continue;

                for (int k = 0; k < observation.second.size(); k++)
                {
                    if (observation.second[k] < 0)
                        continue;

                    // 3D
                    // cv::Vec3f kl = pKFi->mvLineEq[observation.second[k]];
                    // if (kl[0] == 0.0 || kl[0] == -1.0)
                    //     continue;

                    // Eigen::Vector3d line_obs(kl[0], kl[1], kl[2]);

                    // 2D
                    Eigen::Vector3d line_obs;
                    line_obs = pKFi->mvKeyLineFunctions[observation.second[k]];
                    if (line_obs[0] == 0.0 || line_obs[0] == -1.0)
                        continue;

                    if (isnan(line_obs[0]) || isnan(line_obs[1]) || isnan(line_obs[2]))
                    {
                        continue;
                    }

                    // PerpEptsNVector3DMultiFrame *pae = new PerpEptsNVector3DMultiFrame();

                    PerpEptsNVector2DMultiFrame *pae = new PerpEptsNVector2DMultiFrame();

                    pae->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ids)));
                    pae->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(ide)));
                    pae->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                    pae->setMeasurement(line_obs);
                    pae->setInformation(Eigen::Matrix3d::Identity() * (invSigmaStruct + perp_obs_size / 10.0));

                    g2o::RobustKernelHuber *rk_line_e = new g2o::RobustKernelHuber;
                    pae->setRobustKernel(rk_line_e);
                    rk_line_e->setDelta(thHuberLPar);

                    pae->fx = pKF->fx;
                    pae->fy = pKF->fy;
                    pae->cx = pKF->cx;
                    pae->cy = pKF->cy;

                    optimizer.addEdge(pae);
                    vpLineEdgesPerpObs.push_back(pae);
                    vpMapLineEdgePerp.push_back(pML);
                    vpLineEdgeKFPerp.push_back(pKFi);
                    vpLineObsIdxPerp.push_back(k);
                }
            }
        }

        std::sort(MapLineID.begin(), MapLineID.end());
        int maxMapLineID = MapLineID[MapLineID.size() - 1];

        if (pbStopFlag)
            if (*pbStopFlag)
                return;

        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool bDoMore = true;

        if (pbStopFlag)
            if (*pbStopFlag)
                bDoMore = false;
        if (bDoMore)
        {
            // Check inlier observations
            for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
            {
                g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
                MapPoint *pMP = vpMapPointEdgeMono[i];

                if (pMP->isBad())
                    continue;

                if (e->chi2() > 5.991 || !e->isDepthPositive())
                {
                    e->setLevel(1);
                }
                e->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpMapLineEdge.size(); i < iend; i++)
            {
                DistPt2Line2DMultiFrame *e1 = vpLineEdgesSP[i];
                DistPt2Line2DMultiFrame *e2 = vpLineEdgesEP[i];
                MapLine *pML = vpMapLineEdge[i];

                if (pML->isBad())
                    continue;

                if (e1->chi2() > 3.84 || e2->chi2() > 3.84)
                {
                    e1->setLevel(1);
                    e2->setLevel(1);
                }
                e1->setRobustKernel(0);
                e2->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpMapLineEdgePar.size(); i < iend; i++)
            {
                // EdgeLineBinaryEptsTwoD *e1 = vpLineEdgesParObs[i];
                ParEptsNVector2DMultiFrame *e1 = vpLineEdgesParObs[i];

                MapLine *pML = vpMapLineEdgePar[i];

                if (pML->isBad())
                    continue;

                if (e1->chi2() > 0.13)
                {
                    e1->setLevel(1);
                }
                e1->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpMapLineEdgePerp.size(); i < iend; i++)
            {
                // PerpEptsNVector3DMultiFrame* e1 = vpLineEdgesPerpObs[i];
                PerpEptsNVector2DMultiFrame *e1 = vpLineEdgesPerpObs[i];

                MapLine *pML = vpMapLineEdgePerp[i];

                if (pML->isBad())
                    continue;

                if (e1->chi2() > 0.13)
                {
                    e1->setLevel(1);
                }
                e1->setRobustKernel(0);
            }
            // Optimize again without the outliers
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }

        vector<pair<KeyFrame *, MapPoint *>> vToErase;
        vToErase.reserve(vpEdgesMono.size());
        // check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
        {
            g2o::EdgeSE3ProjectXYZ *e = vpEdgesMono[i];
            MapPoint *pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad())
                continue;

            if (e->chi2() > 5.991 || !e->isDepthPositive())
            {
                KeyFrame *pKFi = vpEdgeKFMono[i];
                vToErase.push_back(make_pair(pKFi, pMP));
            }
        }

        vector<pair<KeyFrame *, MapLine *>> vLineToErase;
        vLineToErase.reserve(2 * vpLineEdgesSP.size());
        for (size_t i = 0, iend = vpLineEdgesSP.size(); i < iend; i++)
        {
            DistPt2Line2DMultiFrame *e1 = vpLineEdgesSP[i];
            DistPt2Line2DMultiFrame *e2 = vpLineEdgesSP[i];
            MapLine *pML = vpMapLineEdge[i];

            if (pML->isBad())
                continue;

            if (e1->chi2() > 3.84 || e2->chi2() > 3.84)
            {
                KeyFrame *pKFi = vpLineEdgeKF[i];
                vLineToErase.push_back(make_pair(pKFi, pML));
            }
        }

        int counter_out = 0;
        vector<pair<KeyFrame *, MapLine *>> vLineObsManhToErase;
        vLineObsManhToErase.reserve(vpMapLineManhEdge.size());
        for (size_t i = 0, iend = vpLineManh.size(); i < iend; i++)
        {
            ParEptsNVector3DSingleFrame *e1 = vpLineManh[i];
            MapLine *pML = vpMapLineManhEdge[i];

            if (pML->isBad())
                continue;

            if (e1->chi2() > 0.13)
            {
                KeyFrame *pKFi = vpLineManhKF[i];
                vLineObsManhToErase.push_back(make_pair(pKFi, pML));
                counter_out++;
            }
        }

        vector<pair<MapLine *, pair<KeyFrame *, int>>> vLineObsParToErase;

        vLineObsParToErase.reserve(vpLineEdgesParObs.size());
        for (size_t i = 0, iend = vpLineEdgesParObs.size(); i < iend; i++)
        {
            // ParEptsNVector3DMultiFrame *e1 = vpLineEdgesParObs[i];
            ParEptsNVector2DMultiFrame *e1 = vpLineEdgesParObs[i];
            MapLine *pML = vpMapLineEdgePar[i];

            if (pML->isBad())
                continue;

            if (e1->chi2() > 0.13)
            {
                KeyFrame *pKFi = vpLineEdgeKFPar[i];
                int idx = vpLineObsIdxPar[i];
                std::pair<KeyFrame *, int> kf_n_idx(pKFi, idx);
                vLineObsParToErase.push_back(make_pair(pML, kf_n_idx));
                // pML->EraseParObs(pKFi, idx);
            }
        }

        vector<pair<MapLine *, pair<KeyFrame *, int>>> vLineObsPerpToErase;
        for (size_t i = 0, iend = vpLineEdgesPerpObs.size(); i < iend; i++)
        {
            // PerpEptsNVector3DMultiFrame *e1 = vpLineEdgesPerpObs[i];
            PerpEptsNVector2DMultiFrame *e1 = vpLineEdgesPerpObs[i];

            MapLine *pML = vpMapLineEdgePerp[i];

            if (pML->isBad())
                continue;

            if (e1->chi2() > 0.13)
            {
                KeyFrame *pKFi = vpLineEdgeKFPerp[i];
                int idx = vpLineObsIdxPerp[i];
                std::pair<KeyFrame *, int> kf_n_idx(pKFi, idx);
                vLineObsPerpToErase.push_back(make_pair(pML, kf_n_idx));
                // pML->ErasePerpObs(pKFi, idx);
            }
        }

        // Get Map Mutex
        unique_lock<mutex> lock(pMap->mMutexMapUpdate);
        if (!vToErase.empty())
        {
            for (size_t i = 0; i < vToErase.size(); i++)
            {
                KeyFrame *pKFi = vToErase[i].first;
                MapPoint *pMPi = vToErase[i].second;
                pKFi->EraseMapPointMatch(pMPi);
                pMPi->EraseObservation(pKFi);
            }
        }
        if (!vLineToErase.empty())
        {
            for (size_t i = 0; i < vLineToErase.size(); i++)
            {
                KeyFrame *pKFi = vLineToErase[i].first;
                MapLine *pMLi = vLineToErase[i].second;
                pKFi->EraseMapLineMatch(pMLi);
                pMLi->EraseObservation(pKFi);
            }
        }
        if (!vLineObsManhToErase.empty())
        {
            for (size_t i = 0; i < vLineObsManhToErase.size(); i++)
            {
                KeyFrame *pKFi = vLineObsManhToErase[i].first;
                MapLine *pML = vLineObsManhToErase[i].second;
                pML->EraseManhObs(pKFi);
            }
        }

        if (!vLineObsParToErase.empty())
        {
            for (size_t i = 0; i < vLineObsParToErase.size(); i++)
            {
                MapLine *pML = vLineObsParToErase[i].first;
                KeyFrame *pKFi = vLineObsParToErase[i].second.first;
                int idx = vLineObsParToErase[i].second.second;
                pML->EraseParObs(pKFi, idx);
            }
        }

        if (!vLineObsPerpToErase.empty())
        {
            for (size_t i = 0; i < vLineObsPerpToErase.size(); i++)
            {
                MapLine *pML = vLineObsPerpToErase[i].first;
                KeyFrame *pKFi = vLineObsPerpToErase[i].second.first;
                int idx = vLineObsPerpToErase[i].second.second;
                pML->ErasePerpObs(pKFi, idx);
            }
        }

        // Recover optimized data
        // Keyframes
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            KeyFrame *pKF = *lit;
            g2o::VertexSE3Expmap *vSE3 = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(pKF->mnId));
            g2o::SE3Quat SE3quat = vSE3->estimate();
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        // Points
        for (list<MapPoint *>::iterator lit = lLocalMapPoints.begin(), lend = lLocalMapPoints.end(); lit != lend; lit++)
        {
            MapPoint *pMP = *lit;
            g2o::VertexSBAPointXYZ *vPoint = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(pMP->mnId + maxKFid + 1));
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        // Lines
        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++)
        {
            MapLine *pML = *lit;

            // Check if the vertex exist
            if (optimizer.vertex(2 * pML->mnId + maxMapPointID + 1) == 0)
            {
                continue;
            }

            g2o::VertexSBAPointXYZ *vStartP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(2 * pML->mnId + maxMapPointID + 1));
            g2o::VertexSBAPointXYZ *vEndP = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(2 * pML->mnId + maxMapPointID + 2));

            Vector6d LinePos;
            LinePos << Converter::toVector3d(Converter::toCvMat(vStartP->estimate())), Converter::toVector3d(Converter::toCvMat(vEndP->estimate()));
            pML->SetWorldPos(LinePos);
            pML->UpdateAverageDir();
            pML->UpdateManhAxis();
        }
    }

    void Optimizer::MultiViewManhInit(cv::Mat &manh_axis, KeyFrame *pKF, bool *pbStopFlag, Map *pMap)
    {
        double invSigma = 0.5;
        // Local KeyFrames: First Breath Search from Current KeyFrame
        list<KeyFrame *> lLocalKeyFrames;

        // step1: Add the current keyframe to lLocalKeyFrames
        lLocalKeyFrames.push_back(pKF);
        pKF->mnBALocalForKF = pKF->mnId;

        // step2:Find the keyframe connected by the keyframe (first level connection), and add it to lLocalKeyFrames
        const vector<KeyFrame *> vNeighKFs = pKF->GetVectorCovisibleKeyFrames();
        for (int i = 0, iend = vNeighKFs.size(); i < iend; i++)
        {
            KeyFrame *pKFi = vNeighKFs[i];
            pKFi->mnBALocalForKF = pKF->mnId;
            if (!pKFi->isBad())
                lLocalKeyFrames.push_back(pKFi);
        }

        list<MapLine *> lLocalMapLines;
        // step4: use lLocalKeyFrames to extract the MapLines that can be observed in each key frame, and put them in lLocalMapLines
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            vector<MapLine *> vpMLs = (*lit)->GetMapLineMatches();
            for (vector<MapLine *>::iterator vit = vpMLs.begin(), vend = vpMLs.end(); vit != vend; vit++)
            {
                MapLine *pML = *vit;
                if (pML)
                {
                    if (!pML->isBad())
                    {
                        if (pML->mnBALocalForKF != pKF->mnId)
                        {
                            lLocalMapLines.push_back(pML);
                            pML->mnBALocalForKF = pKF->mnId;
                        }
                    }
                }
            }
        }

        list<KeyFrame *> lFixedCameras;

        // step6:Keyframes that are observed by local MapLines, but do not belong to local keyframes. These keyframes are fixed during local BA optimization
        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++)
        {
            map<KeyFrame *, size_t> observations = (*lit)->GetObservations();
            for (map<KeyFrame *, size_t>::iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
            {
                KeyFrame *pKFi = mit->first;

                if (pKFi->mnBALocalForKF != pKF->mnId && pKFi->mnBAFixedForKF != pKF->mnId)
                {
                    pKFi->mnBAFixedForKF = pKF->mnId;
                    if (!pKFi->isBad())
                        lFixedCameras.push_back(pKFi);
                }
            }
        }

        // step6：Constructor g2o optimizer
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
        linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
        g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false);

        if (pbStopFlag)
            optimizer.setForceStopFlag(pbStopFlag);

        unsigned long maxKFid = 0;
        // step7：Add vertices of the Pose of Local KeyFrames
        for (list<KeyFrame *>::iterator lit = lLocalKeyFrames.begin(), lend = lLocalKeyFrames.end(); lit != lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(pKFi->mnId == 0);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        // step8:Add the vertices of the Pose Fixed KeyFrame
        for (list<KeyFrame *>::iterator lit = lFixedCameras.begin(), lend = lFixedCameras.end(); lit != lend; lit++)
        {
            KeyFrame *pKFi = *lit;
            g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
            vSE3->setEstimate(Converter::toSE3Quat(pKFi->GetPose()));
            vSE3->setId(pKFi->mnId);
            vSE3->setFixed(true);
            optimizer.addVertex(vSE3);
            if (pKFi->mnId > maxKFid)
                maxKFid = pKFi->mnId;
        }

        const float thHuberLEnd = sqrt(1 - 0.997);

        //***********************Set MapLine Vertices******************************
        // step10：Add the vertices of the MapLine
        const int nLineExpectedSize = (lLocalKeyFrames.size() + lFixedCameras.size()) * lLocalMapLines.size();

        vector<Par2Vectors3DMultiFrame *> vpSameManhLineEdges;
        vpSameManhLineEdges.reserve(nLineExpectedSize);

        vector<Perp2Vectors3DMultiFrame *> vpPerpManhLineEdges;
        vpPerpManhLineEdges.reserve(nLineExpectedSize);

        // vector<KeyFrame*> vpLineEdgeKF;
        // vpLineEdgeKF.reserve(nLineExpectedSize);

        vector<MapLine *> vpSameMapLineEdge;
        vpSameMapLineEdge.reserve(nLineExpectedSize);

        vector<MapLine *> vpPerpMapLineEdge;
        vpPerpMapLineEdge.reserve(nLineExpectedSize);

        // Add line vertex
        vector<int> MapLineID;

        for (int idx = 1; idx < 4; idx++)
        {
            cv::Mat eval_axis = manh_axis.col(idx - 1);
            Eigen::Vector3d manh_axis_obs(eval_axis.at<double>(0), eval_axis.at<double>(1), eval_axis.at<double>(2));

            g2o::VertexSBAPointXYZ *vManhAxis = new g2o::VertexSBAPointXYZ();
            vManhAxis->setEstimate(manh_axis_obs);
            int ids = idx + maxKFid;
            vManhAxis->setId(ids);
            vManhAxis->setMarginalized(true);
            optimizer.addVertex(vManhAxis);
        }

        int counter_size = 0;

        for (list<MapLine *>::iterator lit = lLocalMapLines.begin(), lend = lLocalMapLines.end(); lit != lend; lit++)
        {
            MapLine *pML = *lit;
            // Reject the non-associated lines to a Manh. axis
            if (pML->GetManhIdx() == 0)
                continue;

            // This is used to include not only the parallel axis, but also the perp. Manh axis to the constraints. 
            for (int idx = 1; idx < 4; idx++)
            {
                bool same_axis = true;

                if (pML->GetManhIdx() != idx)
                {
                    // same_axis == false;
                    continue;
                }

                counter_size++;

                int num_par_observations = 0;

                const std::map<KeyFrame *, std::vector<int>> parObservations = pML->GetParObservations();
                for (const auto &observation : parObservations)
                {
                    KeyFrame *pKF = observation.first;
                    if (pKF->isBad() || pKF->mnId > maxKFid)
                        continue;

                    for (int i = 0; i < observation.second.size(); i++)
                    {
                        Eigen::Vector3d line_obs;
                        // Use 3D line equation from frame
                        cv::Vec3f kl = pKF->mvLineEq[observation.second[i]];

                        if (kl[0] == 0.0 || kl[0] == -1.0)
                            continue;

                        Eigen::Vector3d threed_line_obs_fr(kl[0], kl[1], kl[2]);

                        Par2Vectors3DMultiFrame *e1 = new Par2Vectors3DMultiFrame();
                        e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(idx + maxKFid)));

                        e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                        e1->setMeasurement(threed_line_obs_fr);
                        e1->setInformation(Eigen::Matrix3d::Identity() * invSigma);
                        g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                        e1->setRobustKernel(rk_line_s);
                        rk_line_s->setDelta(thHuberLEnd);
                        optimizer.addEdge(e1);
                        vpSameManhLineEdges.push_back(e1);

                        for (int idx_perp = 1; idx_perp < 4; idx_perp++)
                        {
                            if (idx == idx_perp)
                                continue;

                            Perp2Vectors3DMultiFrame *e2 = new Perp2Vectors3DMultiFrame();
                            e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(idx_perp + maxKFid)));

                            e2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                            e2->setMeasurement(threed_line_obs_fr);
                            e2->setInformation(Eigen::Matrix3d::Identity() * invSigma);
                            g2o::RobustKernelHuber *rk_line_p = new g2o::RobustKernelHuber;
                            e2->setRobustKernel(rk_line_p);
                            rk_line_p->setDelta(thHuberLEnd);
                            optimizer.addEdge(e2);
                            vpPerpManhLineEdges.push_back(e2);
                            // vpLineEdgeKF.push_back(pKFi);
                            vpPerpMapLineEdge.push_back(pML);
                        }
                    }
                    num_par_observations++;
                }

                int num_perp_observations = 0;
                const std::map<KeyFrame *, std::vector<int>> perpObservations = pML->GetPerpObservations();

                for (const auto &observation : perpObservations)
                {
                    KeyFrame *pKF = observation.first;
                    if (pKF->isBad() || pKF->mnId > maxKFid)
                        continue;

                    for (int i = 0; i < observation.second.size(); i++)
                    {

                        Eigen::Vector3d line_obs;
                        // Use 3D line equation from frame
                        cv::Vec3f kl = pKF->mvLineEq[observation.second[i]];

                        if (kl[0] == 0.0 || kl[0] == -1.0)
                            continue;

                        Eigen::Vector3d threed_line_obs_fr(kl[0], kl[1], kl[2]);

                        Perp2Vectors3DMultiFrame *e2 = new Perp2Vectors3DMultiFrame();
                        e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(idx + maxKFid)));

                        e2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKF->mnId)));
                        e2->setMeasurement(threed_line_obs_fr);
                        e2->setInformation(Eigen::Matrix3d::Identity() * invSigma);
                        g2o::RobustKernelHuber *rk_line_p = new g2o::RobustKernelHuber;
                        e2->setRobustKernel(rk_line_p);
                        rk_line_p->setDelta(thHuberLEnd);
                        optimizer.addEdge(e2);
                        vpPerpManhLineEdges.push_back(e2);
                        // vpLineEdgeKF.push_back(pKFi);
                        vpPerpMapLineEdge.push_back(pML);
                    }

                    num_perp_observations++;
                }

                const map<KeyFrame *, size_t> observations = pML->GetObservations();

                // Set Edges
                int perp = 0;
                int par = 0;

                for (map<KeyFrame *, size_t>::const_iterator mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
                {
                    KeyFrame *pKFi = mit->first;
                    if (!pKFi->isBad())
                    {
                        if (pKFi->vManhAxisIdx.size() == 0)
                            continue;

                        if (pKFi->vManhAxisIdx[mit->second] != pML->GetManhIdx())
                            continue;

                        Eigen::Vector3d line_obs;
                        // Use 3D line equation from frame
                        cv::Vec3f kl = pKFi->mvLineEq[mit->second];

                        if (kl[0] == 0.0 || kl[0] == -1.0)
                            continue;

                        Eigen::Vector3d threed_line_obs_fr(kl[0], kl[1], kl[2]);

                        if (same_axis)
                        {
                            Par2Vectors3DMultiFrame *e1 = new Par2Vectors3DMultiFrame();
                            e1->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(idx + maxKFid)));

                            e1->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                            e1->setMeasurement(threed_line_obs_fr);
                            e1->setInformation(Eigen::Matrix3d::Identity() * invSigma);
                            g2o::RobustKernelHuber *rk_line_s = new g2o::RobustKernelHuber;
                            e1->setRobustKernel(rk_line_s);
                            rk_line_s->setDelta(thHuberLEnd);
                            optimizer.addEdge(e1);
                            vpSameManhLineEdges.push_back(e1);
                            par++;
                        }

                        else
                        {
                            Perp2Vectors3DMultiFrame *e2 = new Perp2Vectors3DMultiFrame();
                            e2->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(idx + maxKFid)));

                            e2->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFi->mnId)));
                            e2->setMeasurement(threed_line_obs_fr);
                            e2->setInformation(Eigen::Matrix3d::Identity() * invSigma);
                            g2o::RobustKernelHuber *rk_line_p = new g2o::RobustKernelHuber;
                            e2->setRobustKernel(rk_line_p);
                            rk_line_p->setDelta(thHuberLEnd);
                            optimizer.addEdge(e2);
                            vpPerpManhLineEdges.push_back(e2);
                            // vpLineEdgeKF.push_back(pKFi);
                            vpPerpMapLineEdge.push_back(pML);

                            perp++;
                        }
                    }
                }
            }
        }

        if (pbStopFlag)
            if (*pbStopFlag)
                return;
        optimizer.initializeOptimization();
        optimizer.optimize(5);

        bool bDoMore = true;

        if (pbStopFlag)
            if (*pbStopFlag)
                bDoMore = false;
        if (bDoMore)
        {
            for (size_t i = 0, iend = vpSameMapLineEdge.size(); i < iend; i++)
            {
                Par2Vectors3DMultiFrame *e1 = vpSameManhLineEdges[i];
                MapLine *pML = vpSameMapLineEdge[i];

                if (pML->isBad())
                    continue;
                if (e1->chi2() > 0.09)
                {
                    e1->setLevel(1);
                }
                e1->setRobustKernel(0);
            }

            for (size_t i = 0, iend = vpPerpMapLineEdge.size(); i < iend; i++)
            {
                Perp2Vectors3DMultiFrame *e1 = vpPerpManhLineEdges[i];
                MapLine *pML = vpPerpMapLineEdge[i];

                if (pML->isBad())
                    continue;
                if (e1->chi2() > 0.09)
                {
                    e1->setLevel(1);
                }
                e1->setRobustKernel(0);
            }
            // Optimize again without the outliers
            optimizer.initializeOptimization(0);
            optimizer.optimize(10);
        }

        // Get Map Mutex

        // Recover optimized data
        std::vector<cv::Mat> matrices;

        g2o::VertexSBAPointXYZ *vSE3_0 = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(1 + maxKFid));
        matrices.push_back(Converter::toCvMat(vSE3_0->estimate()));

        g2o::VertexSBAPointXYZ *vSE3_1 = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(2 + maxKFid));
        matrices.push_back(Converter::toCvMat(vSE3_1->estimate()));

        g2o::VertexSBAPointXYZ *vSE3_2 = static_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(3 + maxKFid));
        matrices.push_back(Converter::toCvMat(vSE3_2->estimate()));

        cv::Mat opt_manh;
        cv::hconcat(matrices, opt_manh);

        opt_manh.convertTo(opt_manh, CV_64F);

        Mat S, U, VT, x_hat, err;
        cv::SVDecomp(opt_manh, S, U, VT, cv::SVD::FULL_UV);
        cv::Mat opt_manh_SVD;
        opt_manh_SVD = U * VT;

        std::cerr << "old_manh : " << manh_axis << std::endl;
        std::cerr << "opt_manh : " << opt_manh << std::endl;
        std::cerr << "opt_manh_SVD : " << opt_manh_SVD << std::endl;

        manh_axis = opt_manh_SVD;
    }

    void Optimizer::OptimizeEssentialGraph(Map *pMap, KeyFrame *pLoopKF, KeyFrame *pCurKF,
                                           const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                           const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                           const map<KeyFrame *, set<KeyFrame *>> &LoopConnections, const bool &bFixScale)
    {
        // Setup optimizer
        g2o::SparseOptimizer optimizer;
        optimizer.setVerbose(false);
        g2o::BlockSolver_7_3::LinearSolverType *linearSolver =
            new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
        g2o::BlockSolver_7_3 *solver_ptr = new g2o::BlockSolver_7_3(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        solver->setUserLambdaInit(1e-16);
        optimizer.setAlgorithm(solver);

        const vector<KeyFrame *> vpKFs = pMap->GetAllKeyFrames();
        const vector<MapPoint *> vpMPs = pMap->GetAllMapPoints();

        const unsigned int nMaxKFid = pMap->GetMaxKFid();

        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vScw(nMaxKFid + 1);
        vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3>> vCorrectedSwc(nMaxKFid + 1);
        vector<g2o::VertexSim3Expmap *> vpVertices(nMaxKFid + 1);

        const int minFeat = 100;

        // Set KeyFrame vertices
        for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
        {
            KeyFrame *pKF = vpKFs[i];
            if (pKF->isBad())
                continue;
            g2o::VertexSim3Expmap *VSim3 = new g2o::VertexSim3Expmap();

            const int nIDi = pKF->mnId;

            LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

            if (it != CorrectedSim3.end())
            {
                vScw[nIDi] = it->second;
                VSim3->setEstimate(it->second);
            }
            else
            {
                Eigen::Matrix<double, 3, 3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
                Eigen::Matrix<double, 3, 1> tcw = Converter::toVector3d(pKF->GetTranslation());
                g2o::Sim3 Siw(Rcw, tcw, 1.0);
                vScw[nIDi] = Siw;
                VSim3->setEstimate(Siw);
            }

            if (pKF == pLoopKF)
                VSim3->setFixed(true);

            VSim3->setId(nIDi);
            VSim3->setMarginalized(false);
            VSim3->_fix_scale = bFixScale;

            optimizer.addVertex(VSim3);

            vpVertices[nIDi] = VSim3;
        }

        set<pair<long unsigned int, long unsigned int>> sInsertedEdges;

        const Eigen::Matrix<double, 7, 7> matLambda = Eigen::Matrix<double, 7, 7>::Identity();

        // Set Loop edges
        for (map<KeyFrame *, set<KeyFrame *>>::const_iterator mit = LoopConnections.begin(), mend = LoopConnections.end(); mit != mend; mit++)
        {
            KeyFrame *pKF = mit->first;
            const long unsigned int nIDi = pKF->mnId;
            const set<KeyFrame *> &spConnections = mit->second;
            const g2o::Sim3 Siw = vScw[nIDi];
            const g2o::Sim3 Swi = Siw.inverse();

            for (set<KeyFrame *>::const_iterator sit = spConnections.begin(), send = spConnections.end(); sit != send; sit++)
            {
                const long unsigned int nIDj = (*sit)->mnId;
                if ((nIDi != pCurKF->mnId || nIDj != pLoopKF->mnId) && pKF->GetWeight(*sit) < minFeat)
                    continue;

                const g2o::Sim3 Sjw = vScw[nIDj];
                const g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;

                optimizer.addEdge(e);

                sInsertedEdges.insert(make_pair(min(nIDi, nIDj), max(nIDi, nIDj)));
            }
        }

        // Set normal edges
        for (size_t i = 0, iend = vpKFs.size(); i < iend; i++)
        {
            KeyFrame *pKF = vpKFs[i];

            const int nIDi = pKF->mnId;

            g2o::Sim3 Swi;

            LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

            if (iti != NonCorrectedSim3.end())
                Swi = (iti->second).inverse();
            else
                Swi = vScw[nIDi].inverse();

            KeyFrame *pParentKF = pKF->GetParent();

            // Spanning tree edge
            if (pParentKF)
            {
                int nIDj = pParentKF->mnId;

                g2o::Sim3 Sjw;

                LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

                if (itj != NonCorrectedSim3.end())
                    Sjw = itj->second;
                else
                    Sjw = vScw[nIDj];

                g2o::Sim3 Sji = Sjw * Swi;

                g2o::EdgeSim3 *e = new g2o::EdgeSim3();
                e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDj)));
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                e->setMeasurement(Sji);

                e->information() = matLambda;
                optimizer.addEdge(e);
            }

            // Loop edges
            const set<KeyFrame *> sLoopEdges = pKF->GetLoopEdges();
            for (set<KeyFrame *>::const_iterator sit = sLoopEdges.begin(), send = sLoopEdges.end(); sit != send; sit++)
            {
                KeyFrame *pLKF = *sit;
                if (pLKF->mnId < pKF->mnId)
                {
                    g2o::Sim3 Slw;

                    LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

                    if (itl != NonCorrectedSim3.end())
                        Slw = itl->second;
                    else
                        Slw = vScw[pLKF->mnId];

                    g2o::Sim3 Sli = Slw * Swi;
                    g2o::EdgeSim3 *el = new g2o::EdgeSim3();
                    el->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pLKF->mnId)));
                    el->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                    el->setMeasurement(Sli);
                    el->information() = matLambda;
                    optimizer.addEdge(el);
                }
            }

            // Covisibility graph edges
            const vector<KeyFrame *> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
            for (vector<KeyFrame *>::const_iterator vit = vpConnectedKFs.begin(); vit != vpConnectedKFs.end(); vit++)
            {
                KeyFrame *pKFn = *vit;
                if (pKFn && pKFn != pParentKF && !pKF->hasChild(pKFn) && !sLoopEdges.count(pKFn))
                {
                    if (!pKFn->isBad() && pKFn->mnId < pKF->mnId)
                    {
                        if (sInsertedEdges.count(make_pair(min(pKF->mnId, pKFn->mnId), max(pKF->mnId, pKFn->mnId))))
                            continue;

                        g2o::Sim3 Snw;

                        LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

                        if (itn != NonCorrectedSim3.end())
                            Snw = itn->second;
                        else
                            Snw = vScw[pKFn->mnId];

                        g2o::Sim3 Sni = Snw * Swi;

                        g2o::EdgeSim3 *en = new g2o::EdgeSim3();
                        en->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(pKFn->mnId)));
                        en->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(nIDi)));
                        en->setMeasurement(Sni);
                        en->information() = matLambda;
                        optimizer.addEdge(en);
                    }
                }
            }
        }

        // Optimize!
        optimizer.initializeOptimization();
        optimizer.optimize(20);

        unique_lock<mutex> lock(pMap->mMutexMapUpdate);

        // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
        for (size_t i = 0; i < vpKFs.size(); i++)
        {
            KeyFrame *pKFi = vpKFs[i];

            const int nIDi = pKFi->mnId;

            g2o::VertexSim3Expmap *VSim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(nIDi));
            g2o::Sim3 CorrectedSiw = VSim3->estimate();
            vCorrectedSwc[nIDi] = CorrectedSiw.inverse();
            Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = CorrectedSiw.translation();
            double s = CorrectedSiw.scale();

            eigt *= (1. / s); //[R t/s;0 1]

            cv::Mat Tiw = Converter::toCvSE3(eigR, eigt);

            pKFi->SetPose(Tiw);
        }

        // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
        for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
        {
            MapPoint *pMP = vpMPs[i];

            if (pMP->isBad())
                continue;

            int nIDr;
            if (pMP->mnCorrectedByKF == pCurKF->mnId)
            {
                nIDr = pMP->mnCorrectedReference;
            }
            else
            {
                KeyFrame *pRefKF = pMP->GetReferenceKeyFrame();
                nIDr = pRefKF->mnId;
            }

            g2o::Sim3 Srw = vScw[nIDr];
            g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

            cv::Mat P3Dw = pMP->GetWorldPos();
            Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
            Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

            cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
            pMP->SetWorldPos(cvCorrectedP3Dw);

            pMP->UpdateNormalAndDepth();
        }
    }

    int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches1, g2o::Sim3 &g2oS12, const float th2, const bool bFixScale)
    {
        g2o::SparseOptimizer optimizer;
        g2o::BlockSolverX::LinearSolverType *linearSolver;

        linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

        g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        optimizer.setAlgorithm(solver);

        // Calibration
        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;

        // Camera poses
        const cv::Mat R1w = pKF1->GetRotation();
        const cv::Mat t1w = pKF1->GetTranslation();
        const cv::Mat R2w = pKF2->GetRotation();
        const cv::Mat t2w = pKF2->GetTranslation();

        // Set Sim3 vertex
        g2o::VertexSim3Expmap *vSim3 = new g2o::VertexSim3Expmap();
        vSim3->_fix_scale = bFixScale;
        vSim3->setEstimate(g2oS12);
        vSim3->setId(0);
        vSim3->setFixed(false);
        vSim3->_principle_point1[0] = K1.at<float>(0, 2);
        vSim3->_principle_point1[1] = K1.at<float>(1, 2);
        vSim3->_focal_length1[0] = K1.at<float>(0, 0);
        vSim3->_focal_length1[1] = K1.at<float>(1, 1);
        vSim3->_principle_point2[0] = K2.at<float>(0, 2);
        vSim3->_principle_point2[1] = K2.at<float>(1, 2);
        vSim3->_focal_length2[0] = K2.at<float>(0, 0);
        vSim3->_focal_length2[1] = K2.at<float>(1, 1);
        optimizer.addVertex(vSim3);

        // Set MapPoint vertices
        const int N = vpMatches1.size();
        const vector<MapPoint *> vpMapPoints1 = pKF1->GetMapPointMatches();
        vector<g2o::EdgeSim3ProjectXYZ *> vpEdges12;
        vector<g2o::EdgeInverseSim3ProjectXYZ *> vpEdges21;
        vector<size_t> vnIndexEdge;

        vnIndexEdge.reserve(2 * N);
        vpEdges12.reserve(2 * N);
        vpEdges21.reserve(2 * N);

        const float deltaHuber = sqrt(th2);

        int nCorrespondences = 0;

        for (int i = 0; i < N; i++)
        {
            if (!vpMatches1[i])
                continue;

            MapPoint *pMP1 = vpMapPoints1[i];
            MapPoint *pMP2 = vpMatches1[i];

            const int id1 = 2 * i + 1;
            const int id2 = 2 * (i + 1);

            const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

            if (pMP1 && pMP2)
            {
                if (!pMP1->isBad() && !pMP2->isBad() && i2 >= 0)
                {
                    g2o::VertexSBAPointXYZ *vPoint1 = new g2o::VertexSBAPointXYZ();
                    cv::Mat P3D1w = pMP1->GetWorldPos();
                    cv::Mat P3D1c = R1w * P3D1w + t1w;
                    vPoint1->setEstimate(Converter::toVector3d(P3D1c));
                    vPoint1->setId(id1);
                    vPoint1->setFixed(true);
                    optimizer.addVertex(vPoint1);

                    g2o::VertexSBAPointXYZ *vPoint2 = new g2o::VertexSBAPointXYZ();
                    cv::Mat P3D2w = pMP2->GetWorldPos();
                    cv::Mat P3D2c = R2w * P3D2w + t2w;
                    vPoint2->setEstimate(Converter::toVector3d(P3D2c));
                    vPoint2->setId(id2);
                    vPoint2->setFixed(true);
                    optimizer.addVertex(vPoint2);
                }
                else
                    continue;
            }
            else
                continue;

            nCorrespondences++;

            // Set edge x1 = S12*X2
            Eigen::Matrix<double, 2, 1> obs1;
            const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
            obs1 << kpUn1.pt.x, kpUn1.pt.y;

            g2o::EdgeSim3ProjectXYZ *e12 = new g2o::EdgeSim3ProjectXYZ();
            e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id2)));
            e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e12->setMeasurement(obs1);
            const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
            e12->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare1);

            g2o::RobustKernelHuber *rk1 = new g2o::RobustKernelHuber;
            e12->setRobustKernel(rk1);
            rk1->setDelta(deltaHuber);
            optimizer.addEdge(e12);

            // Set edge x2 = S21*X1
            Eigen::Matrix<double, 2, 1> obs2;
            const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
            obs2 << kpUn2.pt.x, kpUn2.pt.y;

            g2o::EdgeInverseSim3ProjectXYZ *e21 = new g2o::EdgeInverseSim3ProjectXYZ();

            e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id1)));
            e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
            e21->setMeasurement(obs2);
            float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
            e21->setInformation(Eigen::Matrix2d::Identity() * invSigmaSquare2);

            g2o::RobustKernelHuber *rk2 = new g2o::RobustKernelHuber;
            e21->setRobustKernel(rk2);
            rk2->setDelta(deltaHuber);
            optimizer.addEdge(e21);

            vpEdges12.push_back(e12);
            vpEdges21.push_back(e21);
            vnIndexEdge.push_back(i);
        }

        // Optimize
        optimizer.initializeOptimization();
        optimizer.optimize(5);

        // Check inliers
        int nBad = 0;
        for (size_t i = 0; i < vpEdges12.size(); i++)
        {
            g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
            if (!e12 || !e21)
                continue;

            if (e12->chi2() > th2 || e21->chi2() > th2)
            {
                size_t idx = vnIndexEdge[i];
                vpMatches1[idx] = static_cast<MapPoint *>(NULL);
                optimizer.removeEdge(e12);
                optimizer.removeEdge(e21);
                vpEdges12[i] = static_cast<g2o::EdgeSim3ProjectXYZ *>(NULL);
                vpEdges21[i] = static_cast<g2o::EdgeInverseSim3ProjectXYZ *>(NULL);
                nBad++;
            }
        }

        int nMoreIterations;
        if (nBad > 0)
            nMoreIterations = 10;
        else
            nMoreIterations = 5;

        if (nCorrespondences - nBad < 10)
            return 0;

        // Optimize again only with inliers
        optimizer.initializeOptimization();
        optimizer.optimize(nMoreIterations);

        int nIn = 0;
        for (size_t i = 0; i < vpEdges12.size(); i++)
        {
            g2o::EdgeSim3ProjectXYZ *e12 = vpEdges12[i];
            g2o::EdgeInverseSim3ProjectXYZ *e21 = vpEdges21[i];
            if (!e12 || !e21)
                continue;

            if (e12->chi2() > th2 || e21->chi2() > th2)
            {
                size_t idx = vnIndexEdge[i];
                vpMatches1[idx] = static_cast<MapPoint *>(NULL);
            }
            else
                nIn++;
        }

        // Recover optimized Sim3
        g2o::VertexSim3Expmap *vSim3_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
        g2oS12 = vSim3_recov->estimate();

        return nIn;
    }

} // namespace ORB_SLAM
