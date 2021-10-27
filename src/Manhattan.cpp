#include "Manhattan.h"

using namespace std;

namespace ORB_SLAM2
{
    Manhattan::Manhattan()
    {}

    Manhattan::Manhattan(const cv::Mat &K)
    {
        mK = K;
        mFx = K.at<float>(0, 0);
        mFy = K.at<float>(1, 1);
        mCx = K.at<float>(0, 2);
        mCy = K.at<float>(1, 2);
        mInvFx = 1.0f / mFx;
        mInvFy = 1.0f / mFy;

        // Initial model parameters coarse Manh. axes
        std::vector<int> axis_0{1, 2, 0};
        std::vector<int> axis_1{2, 0, 1};
        std::vector<int> axis_2{0, 1, 2};
        std::vector<std::vector<int>> v_axis{axis_0, axis_1, axis_2};
        mvAxis = v_axis;
        
        // Thresholds
        double deg_th = 3;       
        mCosThPar = cos(deg_th * 0.0174533);
        mCosThPerp = cos((90.0 - deg_th) * 0.0174533);

        double manh_axis_angle_deg = 6.0;
        double th_rad = manh_axis_angle_deg / 180.0 * M_PI;
        mThManh2AxisAngle = std::cos(th_rad);

        // Used to remove rebundancy in the Manhattan Axes Extraction
        std::vector<cv::Mat> rot_poss(24);
        rot_poss[0] = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        rot_poss[1] = (Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, -1);
        rot_poss[2] = (Mat_<double>(3, 3) << -1, 0, 0, 0, 1, 0, 0, 0, -1);
        rot_poss[3] = (Mat_<double>(3, 3) << -1, 0, 0, 0, -1, 0, 0, 0, 1);
        rot_poss[4] = (Mat_<double>(3, 3) << 1, 0, 0, 0, 0, 1, 0, -1, 0);
        rot_poss[5] = (Mat_<double>(3, 3) << 1, 0, 0, 0, 0, -1, 0, 1, 0);
        rot_poss[6] = (Mat_<double>(3, 3) << -1, 0, 0, 0, 0, 1, 0, 1, 0);
        rot_poss[7] = (Mat_<double>(3, 3) << -1, 0, 0, 0, 0, -1, 0, -1, 0);
        rot_poss[8] = (Mat_<double>(3, 3) << 0, 1, 0, 1, 0, 0, 0, 0, -1);
        rot_poss[9] = (Mat_<double>(3, 3) << 0, 1, 0, -1, 0, 0, 0, 0, 1);
        rot_poss[10] = (Mat_<double>(3, 3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);
        rot_poss[11] = (Mat_<double>(3, 3) << 0, -1, 0 - 1, 0, 0, 0, 0, -1);
        rot_poss[12] = (Mat_<double>(3, 3) << 0, 1, 0, 0, 0, 1, 1, 0, 0);
        rot_poss[13] = (Mat_<double>(3, 3) << 0, 1, 0, 0, 0, -1, -1, 0, 0);
        rot_poss[14] = (Mat_<double>(3, 3) << 0, -1, 0, 0, 0, 1, -1, 0, 0);
        rot_poss[15] = (Mat_<double>(3, 3) << 0, -1, 0, 0, 0, -1, 1, 0, 0);
        rot_poss[16] = (Mat_<double>(3, 3) << 0, 0, 1, 1, 0, 0, 0, 1, 0);
        rot_poss[17] = (Mat_<double>(3, 3) << 0, 0, 1, -1, 0, 0, 0, -1, 0);
        rot_poss[18] = (Mat_<double>(3, 3) << 0, 0, -1, 1, 0, 0, 0, -1, 0);
        rot_poss[19] = (Mat_<double>(3, 3) << 0, 0, -1, -1, 0, 0, 0, 1, 0);
        rot_poss[20] = (Mat_<double>(3, 3) << 0, 0, 1, 0, 1, 0, -1, 0, 0);
        rot_poss[21] = (Mat_<double>(3, 3) << 0, 0, 1, 0, -1, 0, 1, 0, 0);
        rot_poss[22] = (Mat_<double>(3, 3) << 0, 0, -1, 0, 1, 0, 1, 0, 0);
        rot_poss[23] = (Mat_<double>(3, 3) << 0, 0, -1, 0, -1, 0, -1, 0, 0);
        mRotPoss = rot_poss;
    }

    // Find those parallel and perpendicular lines
    void Manhattan::computeStructConstrains(Frame &pF, MapLine * pML, std::vector<int> &idxPar, std::vector<int> &idxPerp)
    {
        if (pML->isBad())
            return;
        
        for (size_t i = 0; i < pF.mvLines3D.size(); i++)
        {

            if (pF.mvLines3D[i].first[0] == 0.0)
                continue;
            Eigen::Vector3d line_stp_world = pF.PtToWorldCoord(pF.mvLines3D[i].first);
            Eigen::Vector3d line_ep_world = pF.PtToWorldCoord(pF.mvLines3D[i].second);

            Eigen::Vector3d l_eq_w = line_ep_world - line_stp_world;

            cv::Mat line_eq_world = (Mat_<double>(3, 1) << l_eq_w[0],
                              l_eq_w[1],
                              l_eq_w[2]);
                              
            Vector6d w_line_endpts = pML->GetWorldPos();
            Eigen::Vector3d ml_eq = w_line_endpts.tail(3) - w_line_endpts.head(3);

            cv::Mat pML_eq = (Mat_<double>(3, 1) << ml_eq[0],
                              ml_eq[1],
                              ml_eq[2]);

            double angle = computeAngle(line_eq_world, pML_eq);

            if (abs(angle) < mCosThPerp)
            {
                idxPerp.push_back(i);
            }

            else if (abs(angle) > mCosThPar)
            {
                idxPar.push_back(i);
            }
        }
    }

    void Manhattan::computeStructConstrains(Frame &pF, const int &idx, std::vector<int> &idxPar, std::vector<int> &idxPerp)
    {
        if(pF.mvKeyLineFunctions[idx][2] == 0.0 || pF.mvKeyLineFunctions[idx][2] == -1.0)
            return;
        
        cv::Mat eval_line_eq = (Mat_<double>(3, 1) << double(pF.mvKeyLineFunctions[idx][0]),
                                  double(pF.mvKeyLineFunctions[idx][1]),
                                  double(pF.mvKeyLineFunctions[idx][2]));

        cv::Mat eval_line_eq_3d = (Mat_<double>(3, 1) << double(pF.mvLineEq[idx][0]),
                                  double(pF.mvLineEq[idx][1]),
                                  double(pF.mvLineEq[idx][2]));

        for (int i = 0; i < pF.mvKeyLineFunctions.size(); i++)
        {
            if (i == idx)
                continue;
            cv::Mat line_eq_fr = (Mat_<double>(3, 1) << double(pF.mvKeyLineFunctions[i][0]),
                                  double(pF.mvKeyLineFunctions[i][1]),
                                  double(pF.mvKeyLineFunctions[i][2]));

            cv::Mat eval_line_eq_2d = (Mat_<double>(2, 1) << eval_line_eq.at<double>(0)/ eval_line_eq.at<double>(2),
                                  eval_line_eq.at<double>(1)/ eval_line_eq.at<double>(2));
            
             cv::Mat line_eq_fr_2d = (Mat_<double>(2, 1) << line_eq_fr.at<double>(0)/ line_eq_fr.at<double>(2),
                                  line_eq_fr.at<double>(1)/ line_eq_fr.at<double>(2));
            
            double angle_2d = computeAngle2D(line_eq_fr_2d, eval_line_eq_2d);

            cv::Mat line_eq_fr_3d = (Mat_<double>(3, 1) << double(pF.mvLineEq[i][0]),
                                  double(pF.mvLineEq[i][1]),
                                  double(pF.mvLineEq[i][2]));
            double angle_3d = computeAngle(line_eq_fr_3d, eval_line_eq_3d);

            if (abs(angle_2d) < mCosThPerp 
            && abs(angle_3d) < mCosThPerp
            )
            {
                idxPerp.push_back(i);
            }

            else if (abs(angle_2d) > mCosThPar 
            && abs(angle_3d) > mCosThPar
            )
            {
                idxPar.push_back(i);
            }
        }
    }

    void Manhattan::computeStructConstInMap(Frame &pF, const vector<MapLine *> &mapLines)
    {
        double vert_th = 0.062; // 85 degree
        double par_th = 0.9985; // 5 degree

        for (size_t i = 0; i < pF.mvLineEq.size(); i++)
        {
            cv::Mat line_eq_fr = (Mat_<double>(3, 1) << double(pF.mvLineEq[i][0]),
                                  double(pF.mvLineEq[i][1]),
                                  double(pF.mvLineEq[i][2]));
            cv::Mat line_eq_world = rotCW(line_eq_fr, pF.mTcw);

            int total_lines = 0;
            int num_of_perp_lines = 0;
            int num_of_par_lines = 0;

            std::vector< MapLine*> v_perp_pML;
            std::vector< MapLine*> v_par_pML;

            for (size_t j = 0; j < mapLines.size(); j++)
            {
                MapLine* pML = mapLines[j];

                if (pML->isBad())
                    continue;
                Eigen::Vector3d ml_eq = pML->GetWorldVector();

                cv::Mat pML_eq = (Mat_<double>(3, 1) << ml_eq[0],
                                  ml_eq[1],
                                  ml_eq[2]);

                double angle = computeAngle(line_eq_world, pML_eq);

                total_lines ++;

                if (abs(angle) < vert_th)
                {
                    v_perp_pML.push_back(pML);
                    num_of_perp_lines ++;
                }

                else if (abs(angle) > par_th)
                {
                    v_par_pML.push_back(pML);
                    num_of_par_lines ++;
                }
            }

            pF.mvPerpLines[i] = new std::vector<MapLine*>(v_perp_pML);
            pF.mvParallelLines[i] = new std::vector<MapLine*>(v_par_pML);
        }
    }

    // Transform a line equation from world to camera coordinates
    cv::Mat Manhattan::rotCW(const cv::Mat & lineEq, const cv::Mat Tcw)
    {
        cv::Mat rotcw;
        Tcw.rowRange(0, 3).colRange(0, 3).copyTo(rotcw);

        rotcw.convertTo(rotcw, CV_64F);
        return rotcw * lineEq;
    }

    // Compute Normals and simplify them by computing the main orientations
    void Manhattan::operator()(const cv::Mat &im_depth_resized, const cv::Mat &K,
                               std::vector<cv::Mat> &pt_normals, std::vector<std::vector<cv::Mat>> &main_orients)
    {
        std::vector<cv::Mat> img_normals;
        std::vector<float> depth_normals;
        computeNormalsLPVO(im_depth_resized, K, img_normals, depth_normals);
        // Get the main orientations from an histogram of directions, to reduce the amount of normals used.
        getMainOrientations(img_normals, 10, pt_normals, main_orients);
    }

    void Manhattan::computeNormalsLPVO(const cv::Mat &im_depth_resized, const cv::Mat &cam_K, std::vector<cv::Mat> &pt_normals, std::vector<float> &depth_normals)
    {
        int cell_size = 10;
        // Number of pixels distance between normals extraction. Higher more sparse
        int norm_density = 15;

        // Recover 3D coordinates
        cv::Mat vertexMap = cv::Mat::zeros(im_depth_resized.rows, im_depth_resized.cols, CV_32FC3);
        for (int u = 0; u < im_depth_resized.cols; u++)
        {
            for (int v = 0; v < im_depth_resized.rows; v++)
            {
                float z = im_depth_resized.at<float>(v, u);
                if (z > 0.2f && z < 7.0f)
                {
                    const float x = (u - mCx) * z * mInvFx;
                    const float y = (v - mCy) * z * mInvFy;
                    vertexMap.at<Vec3f>(v, u) = Vec3f(x, y, z);
                }
            }
        }

        // Tangential vectors extraction for each point
        cv::Mat tangeMask = cv::Mat::zeros(im_depth_resized.rows, im_depth_resized.cols, CV_32FC1);

        std::vector<Mat> uTangeMapSplitted{
            cv::Mat::zeros(im_depth_resized.rows, im_depth_resized.cols, CV_32FC1),
            cv::Mat::zeros(im_depth_resized.rows, im_depth_resized.cols, CV_32FC1),
            cv::Mat::zeros(im_depth_resized.rows, im_depth_resized.cols, CV_32FC1)};

        std::vector<Mat> vTangeMapSplitted{
            cv::Mat::zeros(im_depth_resized.rows, im_depth_resized.cols, CV_32FC1),
            cv::Mat::zeros(im_depth_resized.rows, im_depth_resized.cols, CV_32FC1),
            cv::Mat::zeros(im_depth_resized.rows, im_depth_resized.cols, CV_32FC1)};

        for (int u = 1; u < im_depth_resized.cols - 1; u++)
        {
            for (int v = 1; v < im_depth_resized.rows - 1; v++)
            {
                // It evaluates surrounding depth values
                float testCenter = im_depth_resized.at<float>(v, u);
                float testLeft = im_depth_resized.at<float>(v, u - 1);
                float testRight = im_depth_resized.at<float>(v, u + 1);
                float testUp = im_depth_resized.at<float>(v - 1, u);
                float testDown = im_depth_resized.at<float>(v + 1, u);

                if (testCenter < 0.2f || testCenter > 7.0f ||
                    testLeft < 0.2f || testLeft > 7.0f ||
                    testRight < 0.2f || testRight > 7.0f ||
                    testUp < 0.2f || testUp > 7.0f ||
                    testDown < 0.2f || testDown > 7.0f)
                    continue;

                tangeMask.at<float>(v, u) = 1.0f;

                cv::Vec3f u_vertex = vertexMap.at<Vec3f>(v, u + 1) - vertexMap.at<Vec3f>(v, u - 1);
                cv::Vec3f v_vertex = vertexMap.at<Vec3f>(v + 1, u) - vertexMap.at<Vec3f>(v - 1, u);

                uTangeMapSplitted[0].at<float>(v, u) = u_vertex[0];
                uTangeMapSplitted[1].at<float>(v, u) = u_vertex[1];
                uTangeMapSplitted[2].at<float>(v, u) = u_vertex[2];

                vTangeMapSplitted[0].at<float>(v, u) = v_vertex[0];
                vTangeMapSplitted[1].at<float>(v, u) = v_vertex[1];
                vTangeMapSplitted[2].at<float>(v, u) = v_vertex[2];
            }
        }

        // Compute integral image of Tange Map. Delete row and column 0
        cv::Mat u_integral_splitted[3];
        cv::Mat v_integral_splitted[3];

        for (size_t k = 0; k < 3; k++)
        {
            cv::Mat uTangeTemp;
            cv::integral(uTangeMapSplitted[k], uTangeTemp);

            removeMatRow(uTangeTemp, 0);
            removeMatCol(uTangeTemp, 0);
            u_integral_splitted[k] = uTangeTemp;

            cv::Mat vTangeTemp;
            cv::integral(vTangeMapSplitted[k], vTangeTemp);
            removeMatRow(vTangeTemp, 0);
            removeMatCol(vTangeTemp, 0);
            v_integral_splitted[k] = vTangeTemp;
        }

        cv::Mat tangeMaskIntegral;
        cv::integral(tangeMask, tangeMaskIntegral);
        removeMatRow(tangeMaskIntegral, 0);
        removeMatCol(tangeMaskIntegral, 0);

        std::vector<cv::Vec3d> surfaceNormalVector;
        std::vector<cv::Vec2i> surfacePixelPoint;

        cv::Mat surfaceNormalVector1, surfacePixelPoint1;

        for (int v = cell_size; v < im_depth_resized.rows - 1; v += norm_density)
        {
            for (int u = cell_size; u < im_depth_resized.cols - 1; u += norm_density)
            {
                if (tangeMask.at<float>(v, u) == 1)
                {
                    // Average tangential vectors
                    int numPts = tangeMaskIntegral.at<double>(v, u) -
                                 tangeMaskIntegral.at<double>(v - cell_size, u) -
                                 tangeMaskIntegral.at<double>(v, u - cell_size) +
                                 tangeMaskIntegral.at<double>(v - cell_size, u - cell_size);

                    cv::Vec3d uVector((u_integral_splitted[0].at<double>(v, u) -
                                       u_integral_splitted[0].at<double>(v - cell_size, u) -
                                       u_integral_splitted[0].at<double>(v, u - cell_size) +
                                       u_integral_splitted[0].at<double>(v - cell_size, u - cell_size)) /
                                          numPts,
                                      (u_integral_splitted[1].at<double>(v, u) -
                                       u_integral_splitted[1].at<double>(v - cell_size, u) -
                                       u_integral_splitted[1].at<double>(v, u - cell_size) +
                                       u_integral_splitted[1].at<double>(v - cell_size, u - cell_size)) /
                                          numPts,
                                      (u_integral_splitted[2].at<double>(v, u) -
                                       u_integral_splitted[2].at<double>(v - cell_size, u) -
                                       u_integral_splitted[2].at<double>(v, u - cell_size) +
                                       u_integral_splitted[2].at<double>(v - cell_size, u - cell_size)) /
                                          numPts);

                    cv::Vec3d vVector((v_integral_splitted[0].at<double>(v, u) -
                                       v_integral_splitted[0].at<double>(v - cell_size, u) -
                                       v_integral_splitted[0].at<double>(v, u - cell_size) +
                                       v_integral_splitted[0].at<double>(v - cell_size, u - cell_size)) /
                                          numPts,
                                      (v_integral_splitted[1].at<double>(v, u) -
                                       v_integral_splitted[1].at<double>(v - cell_size, u) -
                                       v_integral_splitted[1].at<double>(v, u - cell_size) +
                                       v_integral_splitted[1].at<double>(v - cell_size, u - cell_size)) /
                                          numPts,
                                      (v_integral_splitted[2].at<double>(v, u) -
                                       v_integral_splitted[2].at<double>(v - cell_size, u) -
                                       v_integral_splitted[2].at<double>(v, u - cell_size) +
                                       v_integral_splitted[2].at<double>(v - cell_size, u - cell_size)) /
                                          numPts);

                    cv::Mat normalVector = (Mat_<double>(3, 1) << vVector[1] * uVector[2] - vVector[2] * uVector[1],
                                            vVector[2] * uVector[0] - vVector[0] * uVector[2],
                                            vVector[0] * uVector[1] - vVector[1] * uVector[0]);

                    cv::Mat v_normalized;
                    cv::normalize(normalVector, v_normalized);

                    depth_normals.push_back(float(vertexMap.at<Vec3f>(v, u)[2]));

                    pt_normals.push_back(v_normalized);
                    surfacePixelPoint.push_back(cv::Vec2i(u, v));
                }
            }
        }
    }

    void Manhattan::removeMatRow(cv::Mat &matIn, int row)
    {
        cv::Size size = matIn.size();
        cv::Mat matOut = cv::Mat::zeros(size.height - 1, size.width, matIn.type());

#ifdef USE_CV_RECT
        if (row > 0)
        {
            cv::Rect rect(0, 0, size.width, row);
            matIn(rect).copyTo(matOut(rect));
        }

        if (row < size.height - 1)
        {
            cv::Rect rect1(0, row + 1, size.width, size.height - row - 1);
            cv::Rect rect2(0, row, size.width, size.height - row - 1);
            matIn(rect1).copyTo(matOut(rect2));
        }
#else
        int rowSizeInBytes = size.width * sizeof(float);

        if (row > 0)
        {
            int numRows = row;
            int numBytes = rowSizeInBytes * numRows;
            std::memcpy(matOut.data, matIn.data, numBytes);
        }

        if (row < size.height - 1)
        {
            int matOutOffset = rowSizeInBytes * row;
            int matInOffset = matOutOffset + rowSizeInBytes;

            int numRows = size.height - (row + 1);
            int numBytes = rowSizeInBytes * numRows;
            std::memcpy(matOut.data + matOutOffset, matIn.data + matInOffset, numBytes);
        }
#endif
        matIn = matOut;
    }

    void Manhattan::removeMatCol(cv::Mat &matIn, int col)
    {
        cv::Size size = matIn.size();
        cv::Mat matOut = cv::Mat::zeros(size.height, size.width - 1, matIn.type());

#ifdef USE_CV_RECT
        if (col > 0)
        {
            cv::Rect rect(0, 0, col, size.height);
            matIn(rect).copyTo(matOut(rect));
        }

        if (col < size.width - 1)
        {
            cv::Rect rect1(col + 1, 0, size.width - col - 1, size.height);
            cv::Rect rect2(col, 0, size.width - col - 1, size.height);
            matIn(rect1).copyTo(matOut(rect2));
        }

#else
        int rowInInBytes = size.width * sizeof(float);
        int rowOutInBytes = (size.width - 1) * sizeof(float);

        if (col > 0)
        {
            int matInOffset = 0;
            int matOutOffset = 0;
            int numCols = col;
            int numBytes = numCols * sizeof(float);

            for (int y = 0; y < size.height; ++y)
            {
                std::memcpy(matOut.data + matOutOffset, matIn.data + matInOffset, numBytes);

                matInOffset += rowInInBytes;
                matOutOffset += rowOutInBytes;
            }
        }

        if (col < size.width - 1)
        {
            int matInOffset = (col + 1) * sizeof(float);
            int matOutOffset = col * sizeof(float);
            int numCols = size.width - (col + 1);
            int numBytes = numCols * sizeof(float);

            for (int y = 0; y < size.height; ++y)
            {
                std::memcpy(matOut.data + matOutOffset, matIn.data + matInOffset, numBytes);

                matInOffset += rowInInBytes;
                matOutOffset += rowOutInBytes;
            }
        }
#endif

        matIn = matOut;
    }

    // Filter the values whose vectors orientation have low number of correspondences using a 3D histogram strategy
    void Manhattan::getMainOrientations(const std::vector<cv::Mat> &vec_normals, const int &min_samples, std::vector<cv::Mat> &selected_vectors, std::vector<std::vector<cv::Mat>> &represent_vect)
    {
        // Define the parameters of the Histogram
        int nbins = 16;
        std::pair<float, float> range(-1.0, 1.0);
        int diff_range = range.second - range.first;

        std::vector<cv::Vec3i> v_bins_idx;
        v_bins_idx.resize((vec_normals.size()));

        // Find the bin idx for every axis
        for (size_t i = 0; i < vec_normals.size(); i++)
        {
            int ibinx = (vec_normals[i].at<double>(0) - range.first) * nbins / (diff_range);
            int ibiny = (vec_normals[i].at<double>(1) - range.first) * nbins / (diff_range);
            int ibinz = (vec_normals[i].at<double>(2) - range.first) * nbins / (diff_range);
            v_bins_idx[i] = cv::Vec3i(ibinx, ibiny, ibinz);
        }

        // Join normals with the same axis idx
        std::vector<bool> found(v_bins_idx.size(), false);
        std::vector<std::vector<int>> same_idxs(v_bins_idx.size());
        for (size_t i = 0; i < v_bins_idx.size(); i++)
        {
            if (found[i])
                continue;
            for (size_t j = i; j < v_bins_idx.size(); j++)
            {
                if (found[j])
                    continue;

                if (v_bins_idx[i][0] == v_bins_idx[j][0] || v_bins_idx[i][0] == v_bins_idx[j][0] + 1 || v_bins_idx[i][0] == v_bins_idx[j][0] - 1)
                    if (v_bins_idx[i][1] == v_bins_idx[j][1] || v_bins_idx[i][1] == v_bins_idx[j][1] + 1 || v_bins_idx[i][1] == v_bins_idx[j][1] - 1)
                        if (v_bins_idx[i][2] == v_bins_idx[j][2] || v_bins_idx[i][2] == v_bins_idx[j][2] + 1 || v_bins_idx[i][2] == v_bins_idx[j][2] - 1)
                        {
                            found[j] = true;
                            same_idxs[i].push_back(j);
                        }
            }
        }

        // Reject the associations which size is lower than th.
        for (size_t i = 0; i < same_idxs.size(); i++)
        {
            if (same_idxs[i].size() >= min_samples)
            {
                std::vector<cv::Mat> v_same;
                selected_vectors.push_back(vec_normals[i]);
                for (size_t j = 0; j < same_idxs[i].size(); j++)
                {
                    v_same.push_back(vec_normals[same_idxs[i][j]]);
                    selected_vectors.push_back(vec_normals[same_idxs[i][j]]);
                }
                represent_vect.push_back(v_same);
            }
        }
    }

    // The following function include new features regarding the seekManhattanAxis function (MATLAB) of LPVO - Pyojin Kim
    bool Manhattan::extractCoarseManhAxes(const std::vector<cv::Mat> &v_normals,
                                      const std::vector<cv::Mat> &cand_coord, cv::Mat &manhattan_axis, float &succ_rate)
    {
#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        std::vector<cv::Mat> normal_vector = v_normals;

        if (normal_vector.size() < 20)
            return false;
        // seek Manhattan world frame
        std::vector<cv::Mat> MF_can;
        std::vector<cv::Mat> MF;
        // Minimum of samples to consider the rotation as candidate to refine
        int minSampleNum = round(normal_vector.size() / 35.0);

        int numInitialization = 50;
        int iterNum = 50;
        int c = 20;
        double half_apex_angle_deg = 20.0;

        // Angle variation threshold to stop the refinement process
        double convergeAngleDeg = 0.001;
        double convergeAngle = 0.0174533 * convergeAngleDeg;
        double sin_half_apex_angle = sin(0.0174533 * half_apex_angle_deg);

        // Generate rotation seeds
        for (size_t k = 0; k < cand_coord.size(); k++)
        {
            // cv::Mat R_cM = GetRandomRotation();
            cv::Mat R_cM = cand_coord[k];

            cv::Mat R_cM_update = R_cM.clone();
            cv::Mat R_cM_Rec = Mat::eye(3, 3, CV_64F);
            int numDirectionFound = 0;
            std::vector<int> directionFound;
            int validMF = 0;
            // Evaluates the generated rotations with the line eq. and refine the rot.
            for (int iterCount = 0; iterCount < iterNum; iterCount++)
            {
                R_cM = R_cM_update.clone();
                numDirectionFound = 0;
                directionFound.clear();
                validMF = 0;
                // Use three different main axis configurations
                cv::Mat denTemp = (Mat_<double>(3, 1) << 0.00001, 0.00001, 0.00001);
                for (size_t a = 0; a < mvAxis.size(); a++)
                {
                    cv::Mat r_mat = R_cM.col(mvAxis[a][0]);
                    cv::hconcat(r_mat, R_cM.col(mvAxis[a][1]), r_mat);
                    cv::hconcat(r_mat, R_cM.col(mvAxis[a][2]), r_mat);
                    cv::Mat R_Mc;
                    cv::transpose(r_mat, R_Mc);
                    std::vector<cv::Mat> m_j;
                    // Project to each Manhattan frame axis
                    projectManhAxis(R_Mc, normal_vector, sin_half_apex_angle, m_j);

                    // If a minimum of samples found refine the rotation
                    if (m_j.size() >= minSampleNum)
                    {
                        // Compute mean shift
                        cv::Mat s_j;
                        double density;
                        MeanShift(m_j, c, s_j, density);
                        denTemp.at<double>(a) = density;

                        double alpha = cv::norm(s_j);

                        double tan_alpha = tan(alpha) / alpha;
                        cv::Mat ma_p = (Mat_<double>(3, 1) << tan_alpha * s_j.at<double>(0), tan_alpha * s_j.at<double>(1), 1);
                        cv::Mat R_Mc_transp;
                        cv::transpose(R_Mc, R_Mc_transp);
                        cv::Mat R_cM_Rec_col = R_Mc_transp * ma_p;
                        cv::Mat R_cM_Rec_col_norm = R_cM_Rec_col / cv::norm(R_cM_Rec_col);
                        numDirectionFound++;
                        directionFound.push_back(a);
                        R_cM_Rec.at<double>(0, a) = R_cM_Rec_col_norm.at<double>(0);
                        R_cM_Rec.at<double>(1, a) = R_cM_Rec_col_norm.at<double>(1);
                        R_cM_Rec.at<double>(2, a) = R_cM_Rec_col_norm.at<double>(2);
                    }
                }
                if (numDirectionFound < 2)
                {
                    numDirectionFound = 0;
                    directionFound.clear();
                    validMF = 0;
                    break;
                }

                else if (numDirectionFound == 2)
                {
                    cv::Mat v1 = R_cM_Rec.col(directionFound[0]);
                    cv::Mat v2 = R_cM_Rec.col(directionFound[1]);
                    cv::Mat v3 = v1.cross(v2);

                    int remain_col = 3 - (directionFound[0] + directionFound[1]);
                    R_cM_Rec.at<double>(0, remain_col) = v3.at<double>(0);
                    R_cM_Rec.at<double>(1, remain_col) = v3.at<double>(1);
                    R_cM_Rec.at<double>(2, remain_col) = v3.at<double>(2);

                    // Change vector direction
                    if (abs(cv::determinant(R_cM_Rec) + 1) < 0.5)
                        R_cM_Rec.col(remain_col) *= -1;
                }
                // maintain orthogonality on SO(3) and refine the rotation
                Mat S, U, VT, x_hat, err;
                cv::SVDecomp(R_cM_Rec, S, U, VT, cv::SVD::FULL_UV);
                R_cM_update = U * VT;
                validMF = 1;
                // Trace returns the sum of diagonal elements of matrix
                cv::Mat R_cM_transpose;
                cv::transpose(R_cM, R_cM_transpose);
                cv::Scalar trace_var = trace(R_cM_transpose * R_cM_update);
                double evalAcos = acos((trace_var[0] - 1) / 2);
                // No variance between iterations, it means refined enough
                if (acos((trace_var[0] - 1) / 2) < convergeAngle)
                    break;
            }

            // Found a valid Man. axes
            if (validMF == 1)
            {
                MF.push_back(R_cM_update.clone());
            }
            // Added to accelerate the process
            if (MF.size() > 10)
                break;
        }
        // % check whether we find at least one MF
        if (MF.size() == 0)
        {
            MF_can.clear();
            return false;
        }

        //  Find the unique canonical form
        RemoveRedundancyMF2(MF, MF_can);

        if (MF_can.size() == 0)
            return false;

        float th_ratio = 0.1;

        std::vector<float> v_succ_rate;
        std::vector<cv::Mat> v_manhattan_axis;
        clusterMMF(MF_can, th_ratio, v_manhattan_axis, v_succ_rate);

        // Conditions to return the M_axis
        if (v_manhattan_axis.size() == 1)
        {
            manhattan_axis = v_manhattan_axis[0];
            succ_rate = v_succ_rate[0];
        }
        else
        {
            // Find max succ_rate from a list
            int idx = 0;
            float max_val = v_succ_rate[0];

            for (size_t i = 1; i < v_succ_rate.size(); i++)
            {
                if (v_succ_rate[i] > max_val)
                {
                    idx = i;
                    max_val = v_succ_rate[i];
                }
            }
            manhattan_axis = v_manhattan_axis[idx];
            succ_rate = max_val;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double tMH_axis = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        // std::cerr << "Time to seek the Manh Axis: " << tMH_axis << " Hz: " << 1 / tMH_axis << std::endl;

        return true;
    }

    void Manhattan::projectManhAxis(const cv::Mat &R_Mc, const std::vector<cv::Mat> &normal_vector,
                                    const double &sin_half_apex_angle, std::vector<cv::Mat> &m_j)
    {
        std::vector<cv::Mat> n_j(normal_vector.size());
        // Evaluates the rot with every line eq.
        for (size_t i = 0; i < normal_vector.size(); i++)
        {
            n_j[i] = R_Mc * normal_vector[i];
            double lambda = sqrt(n_j[i].at<double>(0) * n_j[i].at<double>(0) + n_j[i].at<double>(1) * n_j[i].at<double>(1));

            // Evaluates the angle between the rotation and the line eq.
            if (lambda < sin_half_apex_angle)
            {
                double tan_alfa = lambda / abs(n_j[i].at<double>(2));
                double alfa = asin(lambda);

                double s_mj_0 = alfa / tan_alfa * n_j[i].at<double>(0) / n_j[i].at<double>(2);
                double s_mj_1 = alfa / tan_alfa * n_j[i].at<double>(1) / n_j[i].at<double>(2);

                // Filter nan values
                if (s_mj_0 != s_mj_0 && s_mj_1 != s_mj_1)
                    continue;

                m_j.push_back((Mat_<double>(2, 1) << s_mj_0, s_mj_1));
            }
        }
    }

    void Manhattan::MeanShift(const std::vector<cv::Mat> &F, const int &c, cv::Mat &m, double &density)
    {
        cv::Mat nominator = (Mat_<double>(2, 1) << 0, 0);
        double denominator = 0.0;

        for (size_t i = 0; i < F.size(); i++)
        {
            double norm = cv::norm(F[i]);
            double k = exp(-20 * norm * norm);
            nominator.at<double>(0) += k * F[i].at<double>(0);
            nominator.at<double>(1) += k * F[i].at<double>(1);
            denominator += k;
        }

        cv::Mat mm = (Mat_<double>(2, 1) << nominator.at<double>(0) / denominator,
                      nominator.at<double>(1) / denominator);
        m = mm;
        density = denominator / F.size();
    }

    void Manhattan::RemoveRedundancyMF2(const std::vector<cv::Mat> &MF, std::vector<cv::Mat> &MF_can)
    {
        int cell_index = 1;

        for (size_t i = 0; i < MF.size(); i++)
        {
            int minTheta = 100;
            int minID = -1;
            for (size_t j = 0; j < mRotPoss.size(); j++)
            {
                cv::Mat M_star = MF[i] * mRotPoss[j];
                double trace_val = trace(M_star)[0];
                double errTheta = acos((trace_val - 1) / 2);
                if (errTheta < minTheta)
                {
                    minTheta = errTheta;
                    minID = j;
                }
            }
            if (minID != -1)
                MF_can.push_back(MF[i] * mRotPoss[minID]);
        }
    }

    // This function aims at removing the redundancy of the MF (i.e. to achieve
    //  unique canonical) by doing non-maximum suppression.
    void Manhattan::clusterMMF(const std::vector<cv::Mat> &MF_cann, const float &ratio_th, std::vector<cv::Mat> &MF_nonRd, std::vector<float> &succ_rate)
    {
        // Here we utilize a more careful and more smart strategy to find the dominant bins
        std::vector<cv::Mat> MF_can = MF_cann;

        int histStart = 0;
        float histStep = 0.1;
        int histEnd = 2;
        bool HasPeak = true;
        int numMF_can = MF_can.size();
        int numMF = numMF_can;
        int numMF_nonRd = 0;

        while (HasPeak)
        {
            int rnd_idx = rand() % numMF;
            cv::Mat R = MF_can[rnd_idx];
            cv::Mat R_t;
            cv::transpose(R, R_t);
            cv::Mat AA = R_t * R;

            cv::Mat PaaPick;
            cv::Rodrigues(AA, PaaPick);

            cv::Mat Paa(cv::Size(numMF, 3), CV_64FC1, Scalar(0));
            cv::Mat d(cv::Size(numMF, 1), CV_64FC1, Scalar(0));

            for (size_t i = 0; i < numMF; i++)
            {
                cv::Mat Rp = R_t * MF_can[i];

                cv::Mat temp_vect;
                cv::Rodrigues(Rp, temp_vect);

                Paa.at<double>(0, i) = temp_vect.at<double>(0);
                Paa.at<double>(1, i) = temp_vect.at<double>(1);
                Paa.at<double>(2, i) = temp_vect.at<double>(2);

                cv::Mat v_Paa = Paa.col(i);
                d.at<double>(i) = cv::norm(PaaPick - v_Paa);
            }

            std::vector<std::vector<int>> bin;
            EasyHist(d, histStart, histStep, histEnd, bin);

            HasPeak = false;
            for (size_t m = 0; m < bin.size(); m++)
            {
                if (bin[m].size() / float(numMF_can) > ratio_th)
                {
                    HasPeak = true;
                    break;
                }
            }

            if (!HasPeak)
                return;

            // check whether the dominant bin happens at zero
            if (bin[0].size() / float(numMF_can) > ratio_th)
            {
                // calculate the mean, insert the R into the MF_nonRd
                cv::Mat sum = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 0.0);

                for (size_t i = 0; i < bin[0].size(); i++)
                {
                    sum.at<double>(0) += Paa.at<double>(0, bin[0][i]);
                    sum.at<double>(1) += Paa.at<double>(1, bin[0][i]);
                    sum.at<double>(2) += Paa.at<double>(2, bin[0][i]);
                }

                cv::Mat MeanPaaTemp = sum / bin[0].size();
                cv::Mat MeanPaaTemp_t;
                cv::transpose(MeanPaaTemp, MeanPaaTemp_t);
                double angle = cv::norm(MeanPaaTemp);
                cv::Mat aa = MeanPaaTemp_t / angle;
                cv::Mat RTemp;
                cv::Rodrigues(MeanPaaTemp, RTemp);

                // Compensate the rotation matrix
                Mat mask = (RTemp != RTemp);
                int numNaNs = countNonZero(mask);

                if ((numNaNs > 0) && (angle == 0))
                {
                    numMF_nonRd++;
                    MF_nonRd.push_back(R);
                    succ_rate.push_back(bin[0].size() / float(numMF_can));
                }
                else
                {
                    numMF_nonRd++;
                    MF_nonRd.push_back(R * RTemp);
                    succ_rate.push_back(bin[0].size() / float(numMF_can));
                }

                // update MF_can, clean the ones in the found mode
                std::vector<bool> MF_can_selected(MF_can.size(), true);
                std::vector<cv::Mat> MF_can_selection;
                for (size_t j = 0; j < bin[0].size(); j++)
                {
                    MF_can_selected[bin[0][j]] = false;
                }

                for (size_t i = 0; i < MF_can_selected.size(); i++)
                {
                    if (MF_can_selected[i])
                    {
                        MF_can_selection.push_back(MF_can[i]);
                    }
                }

                MF_can = MF_can_selection;
                numMF = MF_can.size();

                if (numMF == 0)
                    return;
            }

            else
                continue;
        }
    }

    void Manhattan::EasyHist(const cv::Mat &data, const int &first, const float &step, const int &last, std::vector<std::vector<int>> &v_bin)
    {
        int numData = data.cols;
        int numBin = (last - first) / step;

        std::vector<std::vector<int>> bin(numBin);

        for (int i = 0; i < numBin; i++)
        {
            float down = (i)*step + first;
            float up = down + step;

            for (int j = 0; j < numData; j++)
            {
                if (data.at<double>(j) >= down && data.at<double>(j) < up)
                    bin[i].push_back(j);
            }
        }
        v_bin = bin;
    }

    void Manhattan::findCoordAxis(const std::vector<std::vector<cv::Mat>> &rep_lines, std::vector<cv::Mat> &v_coord_axis)
    {
        double th_angle_deg = 15;
        double sin_th_angle = sin(0.0174533 * th_angle_deg);

        int counter = 0;

        std::vector<cv::Mat> rep_axis;
        for (size_t i = 0; i < rep_lines.size(); i++)
        {
            //  Use 2D representation
            std::vector<cv::Mat> m_j;
            for (size_t j = 0; j < rep_lines[i].size(); j++)
            {
                double lambda = sqrt(rep_lines[i][j].at<double>(0) * rep_lines[i][j].at<double>(0) + rep_lines[i][j].at<double>(1) * rep_lines[i][j].at<double>(1));

                double tan_alfa = lambda / abs(rep_lines[i][j].at<double>(2));
                double alfa = asin(lambda);

                double s_mj_0 = alfa / tan_alfa * rep_lines[i][j].at<double>(0) / rep_lines[i][j].at<double>(2);
                double s_mj_1 = alfa / tan_alfa * rep_lines[i][j].at<double>(1) / rep_lines[i][j].at<double>(2);

                // Filter nan values
                if (s_mj_0 != s_mj_0 && s_mj_1 != s_mj_1)
                    continue;

                m_j.push_back((Mat_<double>(2, 1) << s_mj_0, s_mj_1));
            }

            // Compute mean shift
            cv::Mat s_j;
            double density;
            MeanShift(m_j, 20, s_j, density);

            // Convert to 3D
            double alpha = cv::norm(s_j);
            double tan_alpha = tan(alpha) / alpha;
            cv::Mat ma_p = (Mat_<double>(3, 1) << tan_alpha * s_j.at<double>(0), tan_alpha * s_j.at<double>(1), 1);

            rep_axis.push_back(ma_p);
            //
        }

        // Compute MeanShift Vector for each one
        // This representative axis is evaluated with other comparing the axis
        for (size_t i = 0; i < rep_axis.size(); i++)
        {
            for (size_t j = i + 1; j < rep_axis.size(); j++)
            {
                double angle = computeAngle(rep_axis[i], rep_axis[j]);
                if (angle < 0.31)
                {
                    cv::Mat s_coord_axis = cv::Mat::zeros(cv::Size(3, 3), CV_64F);

                    //  compute perpendicular axis to this two axes
                    //  The cross product of two non-parallel vectors is a vector that is perpendicular to both of them.
                    cv::Mat third_axis = rep_axis[i].cross(rep_axis[j]);

                    counter++;

                    cv::Mat R_Mc = (cv::Mat::eye(3, 3, CV_64F));
                    cv::Mat R_Mc_transp;

                    cv::Mat r_col0 = rep_axis[i];
                    cv::Mat r_col0_norm = r_col0 / cv::norm(r_col0);

                    cv::Mat r_col1 = rep_axis[j];
                    cv::Mat r_col1_norm = r_col1 / cv::norm(r_col1);

                    cv::Mat r_col2_norm = r_col0.cross(r_col1_norm);

                    s_coord_axis.at<double>(0, 0) = r_col0_norm.at<double>(0);
                    s_coord_axis.at<double>(1, 0) = r_col0_norm.at<double>(1);
                    s_coord_axis.at<double>(2, 0) = r_col0_norm.at<double>(2);

                    s_coord_axis.at<double>(0, 1) = r_col1_norm.at<double>(0);
                    s_coord_axis.at<double>(1, 1) = r_col1_norm.at<double>(1);
                    s_coord_axis.at<double>(2, 1) = r_col1_norm.at<double>(2);

                    s_coord_axis.at<double>(0, 2) = r_col2_norm.at<double>(0);
                    s_coord_axis.at<double>(1, 2) = r_col2_norm.at<double>(1);
                    s_coord_axis.at<double>(2, 2) = r_col2_norm.at<double>(2);

                    // Mantain ortoganility
                    cv::Mat S, U, VT, x_hat, err;
                    cv::SVDecomp(s_coord_axis, S, U, VT, cv::SVD::FULL_UV);
                    cv::Mat ort_coord = U * VT;

                    v_coord_axis.push_back(ort_coord);
                }
            }
        }
    }

    double Manhattan::computeAngle(const cv::Mat &vector1, const cv::Mat &vector2)
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

     double Manhattan::computeAngle2D(const cv::Mat &vector1, const cv::Mat &vector2)
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

    cv::Mat Manhattan::GetRandomRotation()
    {
        // Generate random euler angles
        cv::Mat rand_euler = (Mat_<double>(3, 1) << 0.0174533 * (rand() % 360), 0.0174533 * (rand() % 360), 0.0174533 * (rand() % 360));

        return euler2rot(rand_euler);
    }

    cv::Mat Manhattan::euler2rot(const cv::Mat &euler)
    {
        cv::Mat rotationMatrix(3, 3, CV_64F);

        double x = euler.at<double>(0);
        double y = euler.at<double>(1);
        double z = euler.at<double>(2);

        // Assuming the angles are in radians.
        double ch = cos(z);
        double sh = sin(z);
        double ca = cos(y);
        double sa = sin(y);
        double cb = cos(x);
        double sb = sin(x);

        double m00, m01, m02, m10, m11, m12, m20, m21, m22;

        m00 = ch * ca;
        m01 = sh * sb - ch * sa * cb;
        m02 = ch * sa * sb + sh * cb;
        m10 = sa;
        m11 = ca * cb;
        m12 = -ca * sb;
        m20 = -sh * ca;
        m21 = sh * sa * cb + ch * sb;
        m22 = -sh * sa * sb + ch * cb;

        rotationMatrix.at<double>(0, 0) = m00;
        rotationMatrix.at<double>(0, 1) = m01;
        rotationMatrix.at<double>(0, 2) = m02;
        rotationMatrix.at<double>(1, 0) = m10;
        rotationMatrix.at<double>(1, 1) = m11;
        rotationMatrix.at<double>(1, 2) = m12;
        rotationMatrix.at<double>(2, 0) = m20;
        rotationMatrix.at<double>(2, 1) = m21;
        rotationMatrix.at<double>(2, 2) = m22;
        return rotationMatrix;
    }

    // Assign the associated Manhattan Axis to the extracted lines.
    // Output values: -1-Non-valid line; 0-Non-axis assigned, 1-Axis Manh 0, 2-Axis Manh 1, 3-Axis Manh 2
    void Manhattan::LineManhAxisCorresp(const cv::Mat manh_axis, const std::vector<cv::Vec3f> &v_normals,
                                             std::vector<int> &line_axis)
    {
        
        // // TODO 3 : Avoid this conversion Make this conversion previously because trackManh and SeekManh require this function too
        // // Conversion from cv::Vec3F to cv::Mat
        std::vector<cv::Mat> normal_vector;
        for (size_t i = 0; i < v_normals.size(); i++)
        {
            cv::Mat m_normal = (Mat_<double>(3, 1) << v_normals[i][0],
                                v_normals[i][1],
                                v_normals[i][2]);

            normal_vector.push_back(m_normal);
        }

        std::vector<int> line_axis_corresp(v_normals.size(), 0);

        // // Assign each line to a axis of the Manhattan. Otherwise, keeps the value of 0.
        for (size_t j = 0; j < normal_vector.size(); j++)
        {
            if (normal_vector[j].at<double>(2) == -1.0)
            {
                line_axis_corresp[j] = -1;
                continue;
            }
            for (size_t i = 0; i < 3; i++)
            {
                
                double angle = computeAngle(normal_vector[j], manh_axis.col(i));

                if (angle > mThManh2AxisAngle)
                {
                    line_axis_corresp[j] = i + 1;
                    break;
                }
            }
        }
        line_axis = line_axis_corresp;
    }
   
    // Namespace
}
