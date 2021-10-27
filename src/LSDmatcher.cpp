//
// Created by lan on 17-12-26.
//
#include "LSDmatcher.h"

#define PI 3.1415926

using namespace std;

namespace ORB_SLAM2
{
    const int LSDmatcher::TH_HIGH = 80;
    const int LSDmatcher::TH_LOW = 50;
    const int LSDmatcher::HISTO_LENGTH = 30;

    LSDmatcher::LSDmatcher(float nnratio, bool checkOri) : mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    double LSDmatcher::computeAngle2D(const cv::Mat &vector1, const cv::Mat &vector2)
    {
        // Compute the angle between two vectors
        double dot_product = vector1.dot(vector2);

        // Find magnitude of line AB and BC
        double magnitudeAB = std::sqrt(vector1.at<double>(0) * vector1.at<double>(0) +
                                       vector1.at<double>(1) * vector1.at<double>(1));

        double magnitudeBC = std::sqrt(vector2.at<double>(0) * vector2.at<double>(0) +
                                       vector2.at<double>(1) * vector2.at<double>(1));

        // Find the cosine of the angle formed
        return abs(dot_product / (magnitudeAB * magnitudeBC));
    }

    int LSDmatcher::SearchByGeomNApearance(Frame &CurrentFrame, const Frame &LastFrame, const float desc_th)
    {
        int lmatches = 0;
        std::vector<int> matches_12;

        match(LastFrame.mLdesc, CurrentFrame.mLdesc, desc_th, matches_12);

        std::vector<DMatch> geom_matches;
        const double deltaAngle = M_PI / 8.0;
        const double deltaWidth = (CurrentFrame.mnMaxX - CurrentFrame.mnMinX) * 0.1;
        const double deltaHeight = (CurrentFrame.mnMaxY - CurrentFrame.mnMinY) * 0.1;

        double th_angle = 20.0;
        double th_rad = th_angle / 180.0 * M_PI;
        double cos_th_angle = std::cos(th_rad);

        const int nmatches_12 = matches_12.size();
        for (int i1 = 0; i1 < nmatches_12; ++i1)
        {
            if (!LastFrame.mvpMapLines[i1])
                continue;
            const int i2 = matches_12[i1];
            if (i2 < 0)
                continue;

            if (CurrentFrame.mvKeylinesUn[i2].startPointX == 0)
                continue;

            // check for orientation
            cv::Mat v_line_current = (cv::Mat_<double>(1, 2)
                                          << CurrentFrame.mvKeylinesUn[i2].ePointInOctaveX - CurrentFrame.mvKeylinesUn[i2].sPointInOctaveX,
                                      CurrentFrame.mvKeylinesUn[i2].ePointInOctaveY - CurrentFrame.mvKeylinesUn[i2].sPointInOctaveY);

            cv::Mat v_line_last = (cv::Mat_<double>(1, 2)
                                       << LastFrame.mvKeylinesUn[i1].ePointInOctaveX - LastFrame.mvKeylinesUn[i1].sPointInOctaveX,
                                   LastFrame.mvKeylinesUn[i1].ePointInOctaveY - LastFrame.mvKeylinesUn[i1].sPointInOctaveY);

            double angle = computeAngle2D(v_line_current, v_line_last);

            if (angle < cos_th_angle)
            {
                matches_12[i1] = -1;
                continue;
            }

            // check for position in image
            const float &sX_curr = CurrentFrame.mvKeylinesUn[i2].sPointInOctaveX;
            const float &sX_last = LastFrame.mvKeylinesUn[i1].sPointInOctaveX;
            const float &sY_curr = CurrentFrame.mvKeylinesUn[i2].sPointInOctaveY;
            const float &sY_last = LastFrame.mvKeylinesUn[i1].sPointInOctaveY;
            const float &eX_curr = CurrentFrame.mvKeylinesUn[i2].ePointInOctaveX;
            const float &eX_last = LastFrame.mvKeylinesUn[i1].ePointInOctaveX;
            const float &eY_curr = CurrentFrame.mvKeylinesUn[i2].ePointInOctaveY;
            const float &eY_last = LastFrame.mvKeylinesUn[i1].ePointInOctaveY;

            if ((fabs(sX_curr - sX_last) > deltaWidth || fabs(sY_curr - sY_last) > deltaHeight) && (fabs(eX_curr - eX_last) > deltaWidth || fabs(eY_curr - eY_last) > deltaHeight))
            {
                matches_12[i1] = -1;
                continue;
            }

            cv::DMatch match;
            match.trainIdx = matches_12[i1];
            match.queryIdx = i1;
            match.distance = 2.0;
            geom_matches.push_back(match);
            CurrentFrame.mvpMapLines[i2] = LastFrame.mvpMapLines[i1];
            ++lmatches;
        }

        return lmatches;
    }

    int LSDmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th)
    {
        double th_angle = 10.0;
        double th_rad = th_angle / 180.0 * M_PI;
        double cos_th_angle = std::cos(th_rad);

        int nmatches = 0;
        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH]; // HISTO_LENGTH=30
        for (int i = 0; i < HISTO_LENGTH; i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f / HISTO_LENGTH;

        const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat twc = -Rcw.t() * tcw;

        const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
        const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3);

        const cv::Mat tlc = Rlw * twc + tlw;

        for (int i = 0; i < LastFrame.NL; i++)
        {
            MapLine *pML = LastFrame.mvpMapLines[i];

            if (pML)
            {
                if (!LastFrame.mvbLineOutlier[i])
                {
                    // Project
                    Vector6d Lw = pML->GetWorldPos();
                    if (!CurrentFrame.isInFrustum(pML, 0.5))
                        continue;

                    int nLastOctave = pML->mnTrackScaleLevel;

                    // Search in a window. Size depends on scale
                    float radius = th;

                    vector<size_t> vIndices2;

                    vIndices2 = CurrentFrame.GetFeaturesInAreaForLine(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2,
                                                                      radius, nLastOctave - 1, nLastOctave + 1, 0.96);
                    // vIndices2 = CurrentFrame.GetLinesInArea(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2, radius, nLastOctave-1, nLastOctave+1, 0.96);

                    if (vIndices2.empty())
                        continue;

                    const cv::Mat dML = pML->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
                    {
                        const size_t i2 = *vit;
                        if (CurrentFrame.mvpMapLines[i2])
                            if (CurrentFrame.mvpMapLines[i2]->Observations() > 0)
                                continue;

                        const cv::Mat &d = CurrentFrame.mLdesc.row(i2);

                        // Check orientation
                        cv::Mat v_line_current = (cv::Mat_<double>(1, 2)
                                                      << CurrentFrame.mvKeylinesUn[i2].ePointInOctaveX - CurrentFrame.mvKeylinesUn[i2].sPointInOctaveX,
                                                  CurrentFrame.mvKeylinesUn[i2].ePointInOctaveY - CurrentFrame.mvKeylinesUn[i2].sPointInOctaveY);

                        cv::Mat v_line_last = (cv::Mat_<double>(1, 2)
                                                   << LastFrame.mvKeylinesUn[i].ePointInOctaveX - LastFrame.mvKeylinesUn[i].sPointInOctaveX,
                                               LastFrame.mvKeylinesUn[i].ePointInOctaveY - LastFrame.mvKeylinesUn[i].sPointInOctaveY);

                        double angle = computeAngle2D(v_line_current, v_line_last);

                        if (angle < cos_th_angle)
                            continue;

                        const int dist = DescriptorDistance(dML, d);

                        float max_ = std::max(LastFrame.mvKeylinesUn[i].lineLength, CurrentFrame.mvKeylinesUn[i2].lineLength);
                        float min_ = std::min(LastFrame.mvKeylinesUn[i].lineLength, CurrentFrame.mvKeylinesUn[i2].lineLength);

                        if (min_ / max_ < 0.75)
                            continue;

                        if (dist < bestDist)
                        {
                            bestDist = dist;
                            bestIdx2 = i2;
                        }
                    }

                    if (bestDist <= 95)
                    {
                        CurrentFrame.mvpMapLines[bestIdx2] = pML;
                        nmatches++;
                    }
                }
            }
        }

        return nmatches;
    }

    void LSDmatcher::ComputeThreeMaxima(vector<int> *histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i < L; i++)
        {
            const int s = histo[i].size();
            if (s > max1)
            {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            }
            else if (s > max2)
            {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            }
            else if (s > max3)
            {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2 < 0.1f * (float)max1)
        {
            ind2 = -1;
            ind3 = -1;
        }
        else if (max3 < 0.1f * (float)max1)
        {
            ind3 = -1;
        }
    }

    int LSDmatcher::SearchByProjection(Frame &F, const std::vector<MapLine *> &vpMapLines, const bool eval_orient, const float th)
    {
        int nmatches = 0;

        double th_angle = 15;
        double th_rad = th_angle / 180.0 * M_PI;
        double th_normal = std::cos(th_rad);

        const bool bFactor = th != 1.0;
        for (size_t iML = 0; iML < vpMapLines.size(); iML++)
        {
            MapLine *pML = vpMapLines[iML];

            if (!pML->mbTrackInView)
                continue;

            if (pML->isBad())
                continue;

            const int &nPredictLevel = pML->mnTrackScaleLevel;

            // The size of the window will depend on the viewing direction
            float r = RadiusByViewingCos(pML->mTrackViewCos);

            if (bFactor)
                r *= th;

            // vector<size_t> vIndices = F.GetLinesInArea(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2, r*F.mvScaleFactorsLine[nPredictLevel], nPredictLevel-1, nPredictLevel);

            vector<size_t> vIndices = F.GetFeaturesInAreaForLine(pML->mTrackProjX1, pML->mTrackProjY1, pML->mTrackProjX2, pML->mTrackProjY2, r, nPredictLevel - 1, nPredictLevel);
            if (vIndices.empty())
                continue;
            const cv::Mat MLdescriptor = pML->GetDescriptor();
            Vector6d m_world_pose = pML->GetWorldPos();
            Eigen::Vector3d mWorldVectorML = pML->GetWorldVector();

            int bestDist = 256;
            int bestLevel = -1;
            int bestDist2 = 256;
            int bestLevel2 = -1;
            int bestIdx = -1;

            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                if (F.mvpMapLines[idx])
                    if (F.mvpMapLines[idx]->Observations() > 0)
                        continue;

                const cv::Mat &d = F.mLdesc.row(idx);

                Eigen::Vector3d mWorldVector_frame = F.mvLines3D[idx].first - F.mvLines3D[idx].second;

                float dot = mWorldVector_frame.dot(mWorldVectorML);
                float mag_f = std::sqrt(mWorldVector_frame.x() * mWorldVector_frame.x() + mWorldVector_frame.y() * mWorldVector_frame.y() + mWorldVector_frame.z() * mWorldVector_frame.z());

                float mag_ml = std::sqrt(mWorldVectorML.x() * mWorldVectorML.x() + mWorldVectorML.y() * mWorldVectorML.y() + mWorldVectorML.z() * mWorldVectorML.z());

                float angle = abs(dot / (mag_f * mag_ml));

                if (angle < th_normal)
                    continue;

                const int dist = DescriptorDistance(MLdescriptor, d);

                if (dist < bestDist)
                {
                    bestDist2 = bestDist;
                    bestDist = dist;
                    bestLevel2 = bestLevel;
                    bestLevel = F.mvKeylinesUn[idx].octave;
                    bestIdx = idx;
                }
                else if (dist < bestDist2)
                {
                    bestLevel2 = F.mvKeylinesUn[idx].octave;
                    bestDist2 = dist;
                }
            }

            // Apply ratio to second match (only if best and second are in the same scale level)
            if (bestDist <= 95)
            {
                if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                    continue;

                F.mvpMapLines[bestIdx] = pML;
                nmatches++;
            }
        }
        return nmatches;
    }

    int LSDmatcher::matchNNR(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12)
    {

        int matches = 0;
        matches_12.resize(desc1.rows, -1);

        std::vector<std::vector<cv::DMatch>> matches_;
        cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, false); // cross-check
        bfm->knnMatch(desc1, desc2, matches_, 2);

        if (desc1.rows != matches_.size())
            throw std::runtime_error("[matchNNR] Different size for matches and descriptors!");

        for (int idx = 0, nsize = desc1.rows; idx < nsize; ++idx)
        {
            if (matches_[idx][0].distance < matches_[idx][1].distance * nnr)
            {
                matches_12[idx] = matches_[idx][0].trainIdx;
                matches++;
            }
        }
        return matches;
    }

    int LSDmatcher::match(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12)
    {
        if (false)
        {
            int matches;
            std::vector<int> matches_21;
            // if (true)
            // {
            //     auto match_12 = std::async(std::launch::async, &matchNNR,
            //                                std::cref(desc1), std::cref(desc2), nnr, std::ref(matches_12));
            //     auto match_21 = std::async(std::launch::async, &matchNNR,
            //                                std::cref(desc2), std::cref(desc1), nnr, std::ref(matches_21));
            //     matches = match_12.get();
            //     match_21.wait();
            // }
            // else
            // {
            matches = matchNNR(desc1, desc2, nnr, matches_12);
            matchNNR(desc2, desc1, nnr, matches_21);
            // }

            for (int i1 = 0, nsize = matches_12.size(); i1 < nsize; ++i1)
            {
                int &i2 = matches_12[i1];
                if (i2 >= 0 && matches_21[i2] != i1)
                {
                    i2 = -1;
                    matches--;
                }
            }

            return matches;
        }
        else
            return matchNNR(desc1, desc2, nnr, matches_12);
    }

    int LSDmatcher::SearchDouble(KeyFrame *KF, Frame &CurrentFrame)
    {
        vector<MapLine *> LineMatches = vector<MapLine *>(CurrentFrame.NL, static_cast<MapLine *>(NULL));
        vector<int> tempMatches1 = vector<int>(KF->NL, -1);
        vector<int> tempMatches2 = vector<int>(CurrentFrame.NL, -1);

        Mat ldesc1, ldesc2;
        ldesc1 = KF->mLineDescriptors;
        ldesc2 = CurrentFrame.mLdesc;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        thread thread12(&LSDmatcher::FrameBFMatch, this, ldesc1, ldesc2, std ::ref(tempMatches1), TH_LOW);
        thread thread21(&LSDmatcher::FrameBFMatch, this, ldesc2, ldesc1, std ::ref(tempMatches2), TH_LOW);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches2.size(); i++)
        {
            int j = tempMatches2[i];
            if (j >= 0)
            {
                if (tempMatches1[j] == i)
                {
                    MapLine *pML = KF->GetMapLine(j);
                    if (!pML)
                        continue;
                    CurrentFrame.mvpMapLines[i] = pML;
                    nmatches++;
                }
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchDouble(Frame &InitialFrame, Frame &CurrentFrame, vector<int> &LineMatches)
    {
        LineMatches = vector<int>(InitialFrame.NL, -1);
        vector<int> tempMatches1, tempMatches2;

        Mat ldesc1, ldesc2;
        ldesc1 = InitialFrame.mLdesc;
        ldesc2 = CurrentFrame.mLdesc;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        thread thread12(&LSDmatcher::FrameBFMatch, this, ldesc1, ldesc2, std ::ref(tempMatches1), TH_LOW);
        thread thread21(&LSDmatcher::FrameBFMatch, this, ldesc2, ldesc1, std ::ref(tempMatches2), TH_LOW);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches1.size(); i++)
        {
            int j = tempMatches1[i];
            if (j >= 0)
            {
                if (tempMatches2[j] != i)
                {
                    tempMatches1[i] = -1;
                }
                else
                {
                    nmatches++;
                }
            }
        }

        LineMatches = tempMatches1;

        return nmatches;
    }

    void LSDmatcher::FrameBFMatch(cv::Mat ldesc1, cv::Mat ldesc2, vector<int> &LineMatches, float TH)
    {
        LineMatches = vector<int>(ldesc1.rows, -1);

        vector<vector<DMatch>> lmatches;

        BFMatcher *bfm = new BFMatcher(NORM_HAMMING, false);
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        double nn_dist_th, nn12_dist_th;
        lineDescriptorMAD(lmatches, nn_dist_th, nn12_dist_th);

        nn12_dist_th = nn12_dist_th * 0.5;
        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());

        for (int i = 0; i < lmatches.size(); i++)
        {
            int qdx = lmatches[i][0].queryIdx;
            int tdx = lmatches[i][0].trainIdx;

            double dist_12 = lmatches[i][1].distance - lmatches[i][0].distance;
            if (dist_12 > nn12_dist_th && lmatches[i][0].distance < TH && lmatches[i][0].distance < mfNNratio * lmatches[i][1].distance)
                LineMatches[qdx] = tdx;
        }
    }

    void LSDmatcher::FrameBFMatchNew(cv::Mat ldesc1, cv::Mat ldesc2, vector<int> &LineMatches, vector<KeyLine> kls1, vector<KeyLine> kls2, vector<Eigen::Vector3d> kls2func, cv::Mat F, float TH)
    {
        LineMatches = vector<int>(ldesc1.rows, -1);

        vector<vector<DMatch>> lmatches;

        BFMatcher *bfm = new BFMatcher(NORM_HAMMING, false);
        bfm->knnMatch(ldesc1, ldesc2, lmatches, 2);

        sort(lmatches.begin(), lmatches.end(), sort_descriptor_by_queryIdx());

        for (int i = 0; i < lmatches.size(); i++)
        {
            for (int j = 0; j < lmatches[i].size() - 1; j++)
            {
                int qdx = lmatches[i][j].queryIdx;
                int tdx = lmatches[i][j].trainIdx;

                cv::Mat p1 = (cv::Mat_<float>(3, 1) << kls1[qdx].startPointX, kls1[qdx].startPointY, 1.0);
                cv::Mat p2 = (cv::Mat_<float>(3, 1) << kls1[qdx].endPointX, kls1[qdx].endPointY, 1.0);

                cv::Mat epi_p1 = F * p1;
                cv::Mat epi_p2 = F * p2;

                cv::Mat q1 = (cv::Mat_<float>(3, 1) << kls2[tdx].startPointX, kls2[tdx].startPointY, 1.0);
                cv::Mat q2 = (cv::Mat_<float>(3, 1) << kls2[tdx].endPointX, kls2[tdx].endPointY, 1.0);

                cv::Mat l2 = (cv::Mat_<float>(3, 1) << kls2func[tdx](0), kls2func[tdx](1), kls2func[tdx](2));
                cv::Mat p1_proj = l2.cross(epi_p1);
                cv::Mat p2_proj = l2.cross(epi_p2);

                if (fabs(p1_proj.at<float>(2)) > 1e-12 && fabs(p2_proj.at<float>(2)) > 1e-12)
                {
                    // normalize
                    p1_proj /= p1_proj.at<float>(2);
                    p2_proj /= p2_proj.at<float>(2);

                    std::vector<cv::Mat> collinear_points(4);
                    collinear_points[0] = p1_proj;
                    collinear_points[1] = p2_proj;
                    collinear_points[2] = q1;
                    collinear_points[3] = q2;
                    float score = mutualOverlap(collinear_points);

                    if (lmatches[i][j].distance < TH)
                    {
                        if (score > 0.8 && lmatches[i][j].distance < mfNNratio * lmatches[i][j + 1].distance)
                        {
                            LineMatches[qdx] = tdx;
                            break;
                        }
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    continue;
                }
            }
        }
    }

    float LSDmatcher::mutualOverlap(const std::vector<cv::Mat> &collinear_points)
    {
        float overlap = 0.0f;

        if (collinear_points.size() != 4)
            return 0.0f;

        cv::Mat p1 = collinear_points[0];
        cv::Mat p2 = collinear_points[1];
        cv::Mat q1 = collinear_points[2];
        cv::Mat q2 = collinear_points[3];

        // find outer distance and inner points
        float max_dist = 0.0f;
        size_t outer1 = 0;
        size_t inner1 = 1;
        size_t inner2 = 2;
        size_t outer2 = 3;

        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = i + 1; j < 4; ++j)
            {
                float dist = norm(collinear_points[i] - collinear_points[j]);
                if (dist > max_dist)
                {
                    max_dist = dist;
                    outer1 = i;
                    outer2 = j;
                }
            }
        }

        if (max_dist < 1.0f)
            return 0.0f;

        if (outer1 == 0)
        {
            if (outer2 == 1)
            {
                inner1 = 2;
                inner2 = 3;
            }
            else if (outer2 == 2)
            {
                inner1 = 1;
                inner2 = 3;
            }
            else
            {
                inner1 = 1;
                inner2 = 2;
            }
        }
        else if (outer1 == 1)
        {
            inner1 = 0;
            if (outer2 == 2)
            {
                inner2 = 3;
            }
            else
            {
                inner2 = 2;
            }
        }
        else
        {
            inner1 = 0;
            inner2 = 1;
        }

        overlap = norm(collinear_points[inner1] - collinear_points[inner2]) / max_dist;

        return overlap;
    }

    void LSDmatcher::lineDescriptorMAD(vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const
    {
        vector<vector<DMatch>> matches_nn, matches_12;
        matches_nn = line_matches;
        matches_12 = line_matches;
        // cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;

        // estimate the NN's distance standard deviation
        double nn_dist_median;
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_dist_median = matches_nn[int(matches_nn.size() / 2)][0].distance;

        for (unsigned int i = 0; i < matches_nn.size(); i++)
            matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_mad = 1.4826 * matches_nn[int(matches_nn.size() / 2)][0].distance;

        // estimate the NN's 12 distance standard deviation
        double nn12_dist_median;
        sort(matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
        nn12_dist_median = matches_12[int(matches_12.size() / 2)][1].distance - matches_12[int(matches_12.size() / 2)][0].distance;
        for (unsigned int j = 0; j < matches_12.size(); j++)
            matches_12[j][0].distance = fabsf(matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
        sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
        nn12_mad = 1.4826 * matches_12[int(matches_12.size() / 2)][0].distance;
    }

    int LSDmatcher::DescriptorDistance(const Mat &a, const Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

    int LSDmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           vector<pair<size_t, size_t>> &vMatchedPairs)
    {

        vMatchedPairs.clear();
        vector<int> tempMatches1, tempMatches2;

        Mat ldesc1, ldesc2;
        ldesc1 = pKF1->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        thread thread12(&LSDmatcher::FrameBFMatch, this, ldesc1, ldesc2, std ::ref(tempMatches1), TH_LOW);
        thread thread21(&LSDmatcher::FrameBFMatch, this, ldesc2, ldesc1, std ::ref(tempMatches2), TH_LOW);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches1.size(); i++)
        {
            int j = tempMatches1[i];
            if (j >= 0)
            {
                if (tempMatches2[j] == i)
                {

                    if (pKF1->GetMapLine(i) || pKF2->GetMapLine(j))
                        continue;

                    vMatchedPairs.push_back(make_pair(i, j));
                    nmatches++;
                }
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, vector<int> &vMatchedPairs, bool isDouble)
    {
        vMatchedPairs.clear();
        vMatchedPairs.resize(pKF1->NL, -1);
        vector<int> tempMatches1, tempMatches2;

        Mat ldesc1, ldesc2;
        ldesc1 = pKF1->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        thread thread12(&LSDmatcher::FrameBFMatch, this, ldesc1, ldesc2, std ::ref(tempMatches1), TH_HIGH);
        thread thread21(&LSDmatcher::FrameBFMatch, this, ldesc2, ldesc1, std ::ref(tempMatches2), TH_HIGH);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches1.size(); i++)
        {
            int j = tempMatches1[i];
            if (j >= 0)
            {
                if (isDouble && tempMatches2[j] != i)
                    continue;

                if (pKF1->GetMapLine(i) || pKF2->GetMapLine(j))
                    continue;

                vMatchedPairs[i] = j;
                nmatches++;
            }
        }

        return nmatches;
    }

    int LSDmatcher::SearchForTriangulationNew(KeyFrame *pKF1, KeyFrame *pKF2, vector<int> &vMatchedPairs, bool isDouble)
    {
        vMatchedPairs.clear();
        vMatchedPairs.resize(pKF1->NL, -1);
        vector<int> tempMatches1, tempMatches2;

        Mat ldesc1, ldesc2;
        ldesc1 = pKF1->mLineDescriptors;
        ldesc2 = pKF2->mLineDescriptors;
        if (ldesc1.rows == 0 || ldesc2.rows == 0)
            return 0;

        int nmatches = 0;

        vector<KeyLine> kls1 = pKF1->mvKeyLines;
        vector<KeyLine> kls2 = pKF2->mvKeyLines;
        vector<Eigen::Vector3d> kls1func = pKF1->mvKeyLineFunctions;
        vector<Eigen::Vector3d> kls2func = pKF2->mvKeyLineFunctions;

        cv::Mat F21 = ComputeF12(pKF2, pKF1);
        cv::Mat F12 = ComputeF12(pKF1, pKF2);

        thread thread12(&LSDmatcher::FrameBFMatchNew, this, ldesc1, ldesc2, std ::ref(tempMatches1), kls1, kls2, kls2func, F21, TH_LOW);
        thread thread21(&LSDmatcher::FrameBFMatchNew, this, ldesc2, ldesc1, std ::ref(tempMatches2), kls2, kls1, kls1func, F12, TH_LOW);
        thread12.join();
        thread21.join();

        for (int i = 0; i < tempMatches1.size(); i++)
        {
            int j = tempMatches1[i];
            if (j >= 0)
            {
                if (isDouble && tempMatches2[j] != i)
                    continue;

                if (pKF1->GetMapLine(i) || pKF2->GetMapLine(j))
                    continue;

                vMatchedPairs[i] = j;
                nmatches++;
            }
        }

        return nmatches;
    }

    cv::Mat LSDmatcher::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
    {
        cv::Mat R1w = pKF1->GetRotation();
        cv::Mat t1w = pKF1->GetTranslation();
        cv::Mat R2w = pKF2->GetRotation();
        cv::Mat t2w = pKF2->GetTranslation();

        cv::Mat R12 = R1w * R2w.t();
        cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

        cv::Mat t12x = SkewSymmetricMatrix(t12);

        const cv::Mat &K1 = pKF1->mK;
        const cv::Mat &K2 = pKF2->mK;

        return K1.t().inv() * t12x * R12 * K2.inv();
    }

    int LSDmatcher::Fuse(KeyFrame *pKF, const vector<MapLine *> &vpMapLines, float th)
    {
        cv::Mat Rcw = pKF->GetRotation();
        cv::Mat tcw = pKF->GetTranslation();

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        cv::Mat Ow = pKF->GetCameraCenter();

        int nFused = 0;

        Mat lineDesc = pKF->mLineDescriptors;
        const int nMLs = vpMapLines.size();

        for (int i = 0; i < nMLs; i++)
        {
            MapLine *pML = vpMapLines[i];

            if (!pML)
                continue;

            if (pML->isBad() || pML->IsInKeyFrame(pKF))
                continue;

            Vector6d P = pML->GetWorldPos();

            cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
            cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

            const cv::Mat SPc = Rcw * SP + tcw;
            const float &SPcX = SPc.at<float>(0);
            const float &SPcY = SPc.at<float>(1);
            const float &SPcZ = SPc.at<float>(2);

            const cv::Mat EPc = Rcw * EP + tcw;
            const float &EPcX = EPc.at<float>(0);
            const float &EPcY = EPc.at<float>(1);
            const float &EPcZ = EPc.at<float>(2);

            if (SPcZ < 0.0f || EPcZ < 0.0f)
                return false;

            const float invz1 = 1.0f / SPcZ;
            const float u1 = fx * SPcX * invz1 + cx;
            const float v1 = fy * SPcY * invz1 + cy;

            if (!pKF->IsInImage(u1, v1))
                continue;

            const float invz2 = 1.0f / EPcZ;
            const float u2 = fx * EPcX * invz2 + cx;
            const float v2 = fy * EPcY * invz2 + cy;

            // Depth must be positive
            if (!pKF->IsInImage(u2, v2))
                continue;

            const float maxDistance = pML->GetMaxDistanceInvariance();
            const float minDistance = pML->GetMinDistanceInvariance();

            const cv::Mat OM = 0.5 * (SP + EP) - Ow;
            const float dist = cv::norm(OM);

            if (dist < minDistance || dist > maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Vector3d Pn = pML->GetNormal();
            cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));

            if (OM.dot(pn) < 0.5 * dist)
                continue;

            int nPredictedLevel = pML->PredictScale(dist, pKF->mfLogScaleFactorLine);

            // Search in a radius
            const float radius = th * pKF->mvScaleFactorsLine[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetLinesInArea(u1, v1, u2, v2, radius);

            if (vIndices.empty())
                continue;

            Mat CurrentLineDesc = pML->mLDescriptor; 
            int bestDist = 256;
            int bestIdx = -1;
            for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)
            {
                const size_t idx = *vit;

                const KeyLine &kl = pKF->mvKeyLines[idx];

                const int &kpLevel = kl.octave;

                if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                if (CurrentLineDesc.empty() || dKF.empty())
                    continue;
                const int dist = DescriptorDistance(CurrentLineDesc, dKF);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if (bestDist <= TH_LOW)
            {
                MapLine *pMLinKF = pKF->GetMapLine(bestIdx);

                if (pMLinKF)
                {
                    if (!pMLinKF->isBad())
                    {
                        if (pMLinKF->Observations() > pML->Observations())
                            pML->Replace(pMLinKF);
                        else
                            pMLinKF->Replace(pML);
                    }
                }
                else
                {
                    pML->AddObservation(pKF, bestIdx);
                    pKF->AddMapLine(pML, bestIdx);
                }
                nFused++;
            }
        }
        return nFused;
    }

    float LSDmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if (viewCos > 0.998)
            return 5.0;
        else
            return 8.0;
    }
}
