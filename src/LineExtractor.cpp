#include "LineExtractor.h"
#include <opencv2/line_descriptor/descriptor.hpp>

namespace ORB_SLAM2
{
    LINEextractor::LINEextractor(int _numOctaves, float _scale, unsigned int _nLSDFeature, double _min_line_length) : numOctaves(_numOctaves), scale(_scale), nLSDFeature(_nLSDFeature), min_line_length(_min_line_length)
    {
        mvScaleFactor.resize(numOctaves);
        mvLevelSigma2.resize(numOctaves);
        mvScaleFactor[0] = 1.0f;
        mvLevelSigma2[0] = 1.0f;
        for (int i = 1; i < numOctaves; i++)
        {
            mvScaleFactor[i] = mvScaleFactor[i - 1] * scale;
            mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
        }

        mvInvScaleFactor.resize(numOctaves);
        mvInvLevelSigma2.resize(numOctaves);
        for (int i = 0; i < numOctaves; i++)
        {
            mvInvScaleFactor[i] = 1.0f / mvScaleFactor[i];
            mvInvLevelSigma2[i] = 1.0f / mvLevelSigma2[i];
        }
    }

    double depthStdDev(double d)
    // standard deviation of depth d
    // in meter
    {
        double c1, c2, c3;

        c1 = 0.00273;  // depth_stdev_coeff_c1;
        c2 = 0.00074;  // depth_stdev_coeff_c2;
        c3 = -0.00058; // depth_stdev_coeff_c3;

        return c1 * d * d + c2 * d + c3;
    }

    RandomPoint3d LINEextractor::compPt3dCov(cv::Point3d pt, cv::Mat K, double time_diff_sec)
    {
        RandomPoint3d rp;

        cv::Mat K_double;
        K.convertTo(K_double, CV_64F);

        double f = K_double.at<double>(0, 0), // focal length
            cu = K_double.at<double>(0, 2),
               cv = K_double.at<double>(1, 2);

        //// opencv mat operation is slower than armadillo
        cv::Mat J0 = (cv::Mat_<double>(3, 3) << pt.z / f, 0, pt.x / pt.z,
                      0, pt.z / f, pt.y / pt.z,
                      0, 0, 1);

        cv::Mat cov_g_d0 = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                            0, 1, 0,
                            0, 0, depthStdDev(pt.z) * depthStdDev(pt.z));
        cv::Mat cov0 = J0 * cov_g_d0 * J0.t();

        rp.cov = cov0;
        rp.pos = pt;

        rp.xyz[0] = pt.x;
        rp.xyz[1] = pt.y;
        rp.xyz[2] = pt.z;

        cv::SVD svd(cov0);
        rp.U = svd.u.clone();
        rp.W = svd.w.clone();
        rp.W_sqrt[0] = sqrt(svd.w.at<double>(0));
        rp.W_sqrt[1] = sqrt(svd.w.at<double>(1));
        rp.W_sqrt[2] = sqrt(svd.w.at<double>(2));

        cv::Mat D = (cv::Mat_<double>(3, 3) << 1 / rp.W_sqrt[0], 0, 0,
                     0, 1 / rp.W_sqrt[1], 0,
                     0, 0, 1 / rp.W_sqrt[2]);
        cv::Mat du = D * rp.U.t();
        rp.DU[0] = du.at<double>(0, 0);
        rp.DU[1] = du.at<double>(0, 1);
        rp.DU[2] = du.at<double>(0, 2);
        rp.DU[3] = du.at<double>(1, 0);
        rp.DU[4] = du.at<double>(1, 1);
        rp.DU[5] = du.at<double>(1, 2);
        rp.DU[6] = du.at<double>(2, 0);
        rp.DU[7] = du.at<double>(2, 1);
        rp.DU[8] = du.at<double>(2, 2);
        rp.dux[0] = rp.DU[0] * rp.pos.x + rp.DU[1] * rp.pos.y + rp.DU[2] * rp.pos.z;
        rp.dux[1] = rp.DU[3] * rp.pos.x + rp.DU[4] * rp.pos.y + rp.DU[5] * rp.pos.z;
        rp.dux[2] = rp.DU[6] * rp.pos.x + rp.DU[7] * rp.pos.y + rp.DU[8] * rp.pos.z;

        return rp;
    }

    bool LINEextractor::verify3dLine(const vector<RandomPoint3d> &pts, const cv::Point3d &A, const cv::Point3d &B)
    // input: line AB, collinear points
    // output: whether AB is a good representation for points
    // method: divide AB (or CD, which is endpoints of the projected points on AB)
    // into n sub-segments, detect how many sub-segments containing
    // at least one point(projected onto AB), if too few, then it implies invalid line
    {
        int nCells = 10; // sysPara.num_cells_lineseg_range; // number of cells
        int *cells = new int[nCells];
        double ratio = 0.7; // sysPara.ratio_support_pts_on_line;
        for (int i = 0; i < nCells; ++i)
            cells[i] = 0;
        int nPts = pts.size();
        // find 2 extremities of points along the line direction
        double minv = 100, maxv = -100;
        int idx1 = 0, idx2 = 0;
        for (int i = 0; i < nPts; ++i)
        {
            if ((pts[i].pos - A).dot(B - A) < minv)
            {
                minv = (pts[i].pos - A).dot(B - A);
                idx1 = i;
            }
            if ((pts[i].pos - A).dot(B - A) > maxv)
            {
                maxv = (pts[i].pos - A).dot(B - A);
                idx2 = i;
            }
        }
        cv::Point3d C = projPt3d2Ln3d(pts[idx1].pos, (A + B) * 0.5, B - A);
        cv::Point3d D = projPt3d2Ln3d(pts[idx2].pos, (A + B) * 0.5, B - A);
        double cd = cv::norm(D - C);
        if (cd < 0.0000000001)
        {
            delete[] cells;
            return false;
        }
        for (int i = 0; i < nPts; ++i)
        {
            cv::Point3d X = pts[i].pos;
            double lambda = abs((X - C).dot(D - C) / cd / cd); // 0 <= lambd <=1
            if (lambda >= 1)
            {
                cells[nCells - 1] += 1;
            }
            else
            {
                cells[(unsigned int)floor(lambda * 10)] += 1;
            }
        }
        double sum = 0;
        for (int i = 0; i < nCells; ++i)
        {
            if (cells[i] > 0)
                sum = sum + 1;
        }

        delete[] cells;
        if (sum / nCells > ratio)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    void LINEextractor::computeLine3d_svd(const vector<RandomPoint3d> &pts, const vector<int> &idx, cv::Point3d &mean, cv::Point3d &drct)
    // input: collinear 3d points with noise
    // output: line direction vector and point
    // method: linear equation, PCA
    {
        int n = idx.size();
        mean = cv::Point3d(0, 0, 0);
        for (int i = 0; i < n; ++i)
        {
            mean = mean + pts[idx[i]].pos;
        }
        mean = mean * (1.0 / n);
        cv::Mat P(3, n, CV_64F);
        for (int i = 0; i < n; ++i)
        {
            double pos[3] = {pts[idx[i]].pos.x - mean.x, pts[idx[i]].pos.y - mean.y, pts[idx[i]].pos.z - mean.z};
            array2mat(pos, 3).copyTo(P.col(i));
        }

        cv::SVD svd(P.t(), cv::SVD::MODIFY_A); // FULL_UV is 60 times slower

        drct = mat2cvpt3d(svd.vt.row(0));
    }

    double LINEextractor::mah_dist3d_pt_line(const RandomPoint3d &pt, const cv::Point3d &q1, const cv::Point3d &q2)
    // compute the Mahalanobis distance between a random 3d point p and line (q1,q2)
    // this is fater version since the point cov has already been decomposed by svd
    {
        if (pt.U.cols != 3)
        {
            cerr << "Error in mah_dist3d_pt_line: R matrix must be 3x3" << endl;
            return -1;
        }
        double out;

        double xa = q1.x, ya = q1.y, za = q1.z;
        double xb = q2.x, yb = q2.y, zb = q2.z;
        double c1 = pt.DU[0], c2 = pt.DU[1], c3 = pt.DU[2],
               c4 = pt.DU[3], c5 = pt.DU[4], c6 = pt.DU[5],
               c7 = pt.DU[6], c8 = pt.DU[7], c9 = pt.DU[8];

        double x1 = pt.pos.x, x2 = pt.pos.y, x3 = pt.pos.z;
        double term1 = ((c1 * (x1 - xa) + c2 * (x2 - ya) + c3 * (x3 - za)) * (c4 * (x1 - xb) + c5 * (x2 - yb) + c6 * (x3 - zb)) - (c4 * (x1 - xa) + c5 * (x2 - ya) + c6 * (x3 - za)) * (c1 * (x1 - xb) + c2 * (x2 - yb) + c3 * (x3 - zb))),
               term2 = ((c1 * (x1 - xa) + c2 * (x2 - ya) + c3 * (x3 - za)) * (c7 * (x1 - xb) + c8 * (x2 - yb) + c9 * (x3 - zb)) - (c7 * (x1 - xa) + c8 * (x2 - ya) + c9 * (x3 - za)) * (c1 * (x1 - xb) + c2 * (x2 - yb) + c3 * (x3 - zb))),
               term3 = ((c4 * (x1 - xa) + c5 * (x2 - ya) + c6 * (x3 - za)) * (c7 * (x1 - xb) + c8 * (x2 - yb) + c9 * (x3 - zb)) - (c7 * (x1 - xa) + c8 * (x2 - ya) + c9 * (x3 - za)) * (c4 * (x1 - xb) + c5 * (x2 - yb) + c6 * (x3 - zb))),
               term4 = (c1 * (x1 - xa) - c1 * (x1 - xb) + c2 * (x2 - ya) - c2 * (x2 - yb) + c3 * (x3 - za) - c3 * (x3 - zb)),
               term5 = (c4 * (x1 - xa) - c4 * (x1 - xb) + c5 * (x2 - ya) - c5 * (x2 - yb) + c6 * (x3 - za) - c6 * (x3 - zb)),
               term6 = (c7 * (x1 - xa) - c7 * (x1 - xb) + c8 * (x2 - ya) - c8 * (x2 - yb) + c9 * (x3 - za) - c9 * (x3 - zb));
        out = sqrt((term1 * term1 + term2 * term2 + term3 * term3) / (term4 * term4 + term5 * term5 + term6 * term6));

        return out;
    }

    RandomLine3d LINEextractor::extract3dline_mahdist(const vector<RandomPoint3d> &pts)
    // extract a single 3d line from point clouds using ransac and mahalanobis distance
    // input: 3d points and covariances
    // output: inlier points, line parameters: midpt and direction
    {
        // cout<<"Lineextractor: extract 3d line"<<pts.size()<<endl;
        int maxIterNo = min(10, int(pts.size() * (pts.size() - 1) * 0.5));
        double distThresh = 3.0;
        ; // pt2line_mahdist_extractline; // meter
        // distance threshold should be adapted to line length and depth
        int minSolSetSize = 2;

        vector<int> indexes(pts.size());
        for (size_t i = 0; i < indexes.size(); ++i)
            indexes[i] = i;
        vector<int> maxInlierSet;
        RandomPoint3d bestA, bestB;
        for (int iter = 0; iter < maxIterNo; iter++)
        {
            vector<int> inlierSet;
            random_unique(indexes.begin(), indexes.end(), minSolSetSize); // shuffle
            const RandomPoint3d &A = pts[indexes[0]];

            const RandomPoint3d &B = pts[indexes[1]];

            if (cv::norm(B.pos - A.pos) < 0.0000000001)
                continue;
            for (size_t i = 0; i < pts.size(); ++i)
            {
                // compute distance to AB
                double dist = mah_dist3d_pt_line(pts[i], A.pos, B.pos);
                // cout<<"dist takes "<<dist<<endl;
                // cout<<"dist:A pos"<<A.pos.x<<","<<A.xyz[2]<<endl;
                // cout<<"Lineextractor: dist"<<dist<<endl;
                if (dist < distThresh)
                {
                    inlierSet.push_back(i);
                }
            }
            if (inlierSet.size() > maxInlierSet.size())
            {
                vector<RandomPoint3d> inlierPts(inlierSet.size());
                for (size_t ii = 0; ii < inlierSet.size(); ++ii)
                    inlierPts[ii] = pts[inlierSet[ii]];
                if (verify3dLine(inlierPts, A.pos, B.pos))
                {
                    maxInlierSet = inlierSet;
                    bestA = pts[indexes[0]];
                    bestB = pts[indexes[1]];
                }
            }
            if (maxInlierSet.size() > pts.size() * 0.6)
                break;
        }
        RandomLine3d rl;
        if (maxInlierSet.size() >= 2)
        {
            cv::Point3d m = (bestA.pos + bestB.pos) * 0.5, d = bestB.pos - bestA.pos;
            // optimize and reselect inliers
            // compute a 3d line using algebraic method
            while (true)
            {
                vector<int> tmpInlierSet;
                cv::Point3d tmp_m, tmp_d;
                computeLine3d_svd(pts, maxInlierSet, tmp_m, tmp_d);
                for (size_t i = 0; i < pts.size(); ++i)
                {
                    if (mah_dist3d_pt_line(pts[i], tmp_m, tmp_m + tmp_d) < distThresh)
                    {
                        tmpInlierSet.push_back(i);
                    }
                }
                if (tmpInlierSet.size() > maxInlierSet.size())
                {
                    maxInlierSet = tmpInlierSet;
                    m = tmp_m;
                    d = tmp_d;
                }
                else
                    break;
            }
            // find out two endpoints
            double minv = 100, maxv = -100;
            int idx_end1 = 0, idx_end2 = 0;
            for (size_t i = 0; i < maxInlierSet.size(); ++i)
            {
                double dproduct = (pts[maxInlierSet[i]].pos - m).dot(d);
                if (dproduct < minv)
                {
                    minv = dproduct;
                    idx_end1 = i;
                }
                if (dproduct > maxv)
                {
                    maxv = dproduct;
                    idx_end2 = i;
                }
            }
            rl.A = pts[maxInlierSet[idx_end1]].pos;
            rl.B = pts[maxInlierSet[idx_end2]].pos;
        }
        rl.director = (rl.A - rl.B) / sqrt((rl.A - rl.B).dot(rl.A - rl.B));
        rl.mid = (rl.A + rl.B) / 2;
        rl.pts.resize(maxInlierSet.size());
        for (size_t i = 0; i < maxInlierSet.size(); ++i)
            rl.pts[i] = pts[maxInlierSet[i]];
        return rl;
    }

    void LINEextractor::operator()(cv::InputArray _image, cv::InputArray _mask, std::vector<KeyLine> &_keylines, cv::OutputArray _descriptors, std::vector<Eigen::Vector3d> &_lineVec2d)
    {
        if (_image.empty())
            return;

        Mat image = _image.getMat();
        assert(image.type() == CV_8UC1);

        Mat mask = _mask.getMat();

        // detect line feature
        Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
        lsd->detect(image, _keylines, scale, numOctaves, mask);

        cv::Mat descriptors;

        // filter lines
        if (_keylines.size() > nLSDFeature)
        {
            sort(_keylines.begin(), _keylines.end(), sort_lines_by_response());
            _keylines.resize(nLSDFeature);
            for (unsigned int i = 0; i < nLSDFeature; i++)
                _keylines[i].class_id = i;
        }
        Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
        lbd->compute(image, _keylines, descriptors);

        _lineVec2d.clear();
        for (vector<KeyLine>::iterator it = _keylines.begin(); it != _keylines.end(); ++it)
        {
            Eigen::Vector3d sp_l;
            sp_l << it->startPointX, it->startPointY, 1.0;
            Eigen::Vector3d ep_l;
            ep_l << it->endPointX, it->endPointY, 1.0;
            Eigen::Vector3d lineV; 
            lineV << sp_l.cross(ep_l);
            lineV = lineV / sqrt(lineV(0) * lineV(0) + lineV(1) * lineV(1));
            _lineVec2d.push_back(lineV);
        }

        descriptors.copyTo(_descriptors);
    }

    

    bool LINEextractor::computeBest3dLineRepr(const cv::Mat &rgb_image, const cv::Mat &depth_img, const line_descriptor::KeyLine &keyline, const cv::Mat mK, std::pair<cv::Point, cv::Point> &pair_pts_2D, std::pair<cv::Point3f, cv::Point3f> &end_pts3D, cv::Vec3f &line_vector)
    {
        std::vector<cv::Point3f> threeD_points;

        getPointsFromLineAndUnproject(rgb_image, depth_img, mK, keyline, threeD_points);

        if (threeD_points.size() < 10)
            return false;

        // Compute Endpoints and line vector using a clustering strategy and a line refinement
        if (computeEndpoints(threeD_points, end_pts3D, line_vector))
            return true;
        else
            return false;
    }

    void LINEextractor::getPointsFromLineAndUnproject(const cv::Mat image, const cv::Mat depth_image, const cv::Mat mk, const line_descriptor::KeyLine &line, std::vector<cv::Point3f> &line_points)
    {
        cv::Point start_point(line.startPointX, line.startPointY);
        cv::Point end_point(line.endPointX, line.endPointY);

        cv::LineIterator it(image, start_point, end_point, 8);
        for (int j = 0; j < it.count; j++, ++it)
        {
            float depth_value = depth_image.at<float>(it.pos());
            // // Unproject the point if it is valid
            if ((depth_value > 0.2) && (std::isfinite(depth_value)) && depth_value < 8.0)
            {
                float x = ((it.pos().x - mk.at<float>(0, 2)) * depth_value) / mk.at<float>(0, 0);
                float y = ((it.pos().y - mk.at<float>(1, 2)) * depth_value) / mk.at<float>(1, 1);
                cv::Point3f point(x, y, depth_value);
                line_points.push_back(point);
            }
        }
    }

    bool LINEextractor::computeEndpoints(const std::vector<cv::Point3f> &threeD_points, std::pair<cv::Point3f, cv::Point3f> &end_pts, cv::Vec3f &line_vector)
    {
        // Th normal difference between consecutive points (m)
        float th_normals = 0.2;
        float th_angle = 5.0;

        // Cluster points by means of the evaluation of the normals
        float prev_norm = 0;
        std::vector<std::vector<cv::Point3f>> dist_clusters;
        std::vector<cv::Point3f> single_cluster;

        // Resize the number of samples
        int step = 2;
        if (threeD_points.size() > 120)
            step = int(threeD_points.size() / 30);
        for (size_t i = step; i < threeD_points.size() - step; i += step)
        {
            // TODO 10: use prev_normal instead of compute it each time
            float prev_normal = (threeD_points[i].z - threeD_points[i - step].z) / 2.0;
            float next_normal = (threeD_points[i + step].z - threeD_points[i].z) / 2.0;
            float diff = abs(prev_normal - next_normal);

            // Add point to current cluster
            if (abs(diff) < th_normals)
            {
                if (single_cluster.size() == 0)
                {
                    single_cluster.push_back(threeD_points[i - step]);
                }
                single_cluster.push_back(threeD_points[i]);
            }
            // Create new cluster
            else
            {
                if (single_cluster.size() == 0)
                    single_cluster.push_back(threeD_points[i]);
                dist_clusters.push_back(single_cluster);
                single_cluster.clear();
            }

            prev_norm = next_normal;
        }

        if (dist_clusters.size() == 0)
        {
            if (single_cluster.size() > 10)
            {
                // Improve the accuracy of the endpoints using M-estimators
                Vec6f line_para;
                // Extract the line vector and a point of the line
                fitLine(single_cluster, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
                // Compute the closest point between the line and the old endpoint which corresponds to the line.
                cv::Point3f st_pt = projectionPt2Line(single_cluster[0], line_para);
                cv::Point3f end_pt = projectionPt2Line(single_cluster[single_cluster.size() - 2], line_para);

                end_pts.first = st_pt;
                end_pts.second = end_pt;
                line_vector = cv::Vec3f(line_para[0], line_para[1], line_para[2]);

                return true;
            }
            else
                return false;
        }

        else if (dist_clusters.size() > 0 && single_cluster.size() > 0)
        {
            dist_clusters.push_back(single_cluster);
        }

        // If exist more than two clusters try to merge them
        std::vector<std::vector<int>> merge_idx(dist_clusters.size());
        if (dist_clusters.size() > 1)
        {
            double th_rad = th_angle / 180.0 * M_PI;
            double th_normal = std::cos(th_rad);

            for (size_t i = 0; i < dist_clusters.size(); i++)
            {
                if (dist_clusters[i].size() < 10)
                    continue;

                for (size_t j = i + 1; j < dist_clusters.size(); j++)
                {
                    if (dist_clusters[j].size() < 10)
                        continue;

                    float angle = computeAngle(dist_clusters[i][1], dist_clusters[i][dist_clusters[i].size() - 2], dist_clusters[j][1], dist_clusters[j][dist_clusters[j].size() - 2]);

                    if (angle > th_normal)
                    {
                        merge_idx[i].push_back(j);
                    }
                }
            }

            for (size_t i = 0; i < merge_idx.size(); i++)
            {
                if (merge_idx[i].size() > 0)
                {
                    for (size_t j = 0; j < merge_idx[i].size(); j++)
                    {
                        dist_clusters[i].insert(dist_clusters[i].end(), dist_clusters[merge_idx[i][j]].begin(), dist_clusters[merge_idx[i][j]].end());
                        dist_clusters[merge_idx[i][j]] = dist_clusters[i];
                    }
                }
            }

            int max_size = 0;
            int idx = -1;

            // Find bigger cluster and get the 3D endpoints
            for (size_t i = 0; i < dist_clusters.size(); i++)
            {
                if (dist_clusters[i].size() > max_size)
                {
                    max_size = dist_clusters[i].size();
                    idx = i;
                }
            }

            if (max_size < 10)
                return false;

            // Improve the accuracy of the endpoints using M-estimators
            Vec6f line_para;
            // Extract the line vector and a pt of the line
            fitLine(dist_clusters[idx], line_para, cv::DIST_L2, 0, 1e-2, 1e-2);

            // Compute the closest point between the line and the old endpoint which corresponds to the line.
            cv::Point3f st_pt = projectionPt2Line(dist_clusters[idx][0], line_para);
            cv::Point3f end_pt = projectionPt2Line(dist_clusters[idx][dist_clusters.size() - 2], line_para);

            end_pts.first = st_pt;
            end_pts.second = end_pt;
            line_vector = cv::Vec3f(line_para[0], line_para[1], line_para[2]);

            return true;
        }
    }

    cv::Point3f LINEextractor::projectionPt2Line(const cv::Point3f &point, const cv::Vec6f &vector_n_pt)
    {
        cv::Point3f AB(vector_n_pt[0], vector_n_pt[1], vector_n_pt[2]);
        cv::Point3f A = cv::Point3f(vector_n_pt[3], vector_n_pt[4], vector_n_pt[5]);
        cv::Point3f AM = point - A;
        cv::Point3f P = A + AM.dot(AB) / AB.dot(AB) * AB;
        return P;
    }

    float LINEextractor::computeAngle(const cv::Point3f &pt1, const cv::Point3f &pt2, const cv::Point3f &pt3, const cv::Point3f &pt4)
    {
        // Compute lines using two 3D endpoints of each line-segment
        cv::Point3f v_line_12 = (pt2 - pt1);
        cv::Point3f v_line_34 = (pt4 - pt3);

        // Compute the angle between two line-segments
        double dotProduct = v_line_12.x * v_line_34.x + v_line_12.y * v_line_34.y + v_line_12.z * v_line_34.z;

        // Find magnitude of line AB and BC
        double magnitudeAB = std::sqrt(v_line_12.x * v_line_12.x + v_line_12.y * v_line_12.y + v_line_12.z * v_line_12.z);
        double magnitudeBC = std::sqrt(v_line_34.x * v_line_34.x + v_line_34.y * v_line_34.y + v_line_34.z * v_line_34.z);

        // Find the cosine of the angle formed by line AB and BC
        double angle = dotProduct;
        return abs(dotProduct / (magnitudeAB * magnitudeBC));
    }

    double LINEextractor::computeAngle(const cv::Mat &vector1, const cv::Mat &vector2)
    {
        // Compute the angle between two vectors
        double dot_product = vector1.dot(vector2);

        // Find magnitude of line AB and BC
        double magnitudeAB = std::sqrt(vector1.at<double>(0) * vector1.at<double>(0) +
                                       vector1.at<double>(1) * vector1.at<double>(1) +
                                       vector1.at<double>(2) * vector1.at<double>(2));

        double magnitudeBC = std::sqrt(vector2.at<double>(0) * vector2.at<double>(0) +
                                       vector2.at<double>(1) * vector2.at<double>(1) +
                                       vector2.at<double>(2) * vector2.at<double>(2));

        // Find the cosine of the angle formed
        return abs(dot_product / (magnitudeAB * magnitudeBC));
    }
} // namespace ORB_SLAM2