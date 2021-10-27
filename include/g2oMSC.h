#ifndef g2o_MSC_H
#define g2o_MSC_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/base_multi_edge.h"

namespace g2o
{
    inline double ComputeAngle2D(const Vector3d &l_a_hom, const Vector2d &l_b)
    {
        // Compute the angle between two vectors
        Eigen::Vector2d l_a(l_a_hom[0] / l_a_hom[2], l_a_hom[1] / l_a_hom[2]);

        double dot_product = l_b.dot(l_a);

        // Find magnitude of line AB and BC
        double magn_a = l_b.norm();
        double magn_b = l_a.norm();

        // Find the cosine of the angle formed
        return abs(dot_product / (magn_a * magn_b));
    }

    inline double ComputeAngle3D(const Vector3d &l_a, const Vector3d &l_b)
    {
        double dot_product = l_b.dot(l_a);

        // Find magnitude of lines
        double magn_a = l_b.norm();
        double magn_b = l_a.norm();

        // Find the cosine of the angle formed
        return abs(dot_product / (magn_a * magn_b));
    }

    inline Vector2d project2d(const Vector3d &pt)
    {
        Vector2d res;
        res(0) = pt(0) / pt(2);
        res(1) = pt(1) / pt(2);
        return res;
    }

    inline Vector2d cam_project(const Vector3d &pt, const double &fx, const double &fy, const double &cx, const double &cy)
    {
        Vector2d proj_pt = project2d(pt);
        Vector2d result;
        result[0] = proj_pt[0] * fx + cx;
        result[1] = proj_pt[1] * fy + cy;
        return result;
    }


    class ParEptsNVector3DSingleFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        ParEptsNVector3DSingleFrame() {}

        virtual void computeError()
        {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);

            Vector3d line_eq = end_pt->estimate() - st_pt->estimate();
            double error_angle = ComputeAngle3D(line_eq, _measurement);

            _error(0) = 1 - error_angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class PerpEptsNVector3DSingleFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        PerpEptsNVector3DSingleFrame() {}

        virtual void computeError()
        {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);

            // Extract 3D line eq. from two endpoints
            Vector3d line_eq = end_pt->estimate() - st_pt->estimate();

            double error_angle = ComputeAngle3D(line_eq, _measurement);

            _error(0) = error_angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class ParEptsNVector2DSingleFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        ParEptsNVector2DSingleFrame() {}

        virtual void computeError()
        {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);

            // Project 3D endpoints to image coordinates
            Vector2d proj_st_pt = cam_project(st_pt->estimate(), fx, fy, cx, cy);
            Vector2d proj_end_pt = cam_project(end_pt->estimate(), fx, fy, cx, cy);

            // Extract 2D line equation
            Vector2d line_eq = proj_end_pt - proj_st_pt;

            double angle = ComputeAngle2D(_measurement, line_eq);

            _error(0) = 1 - angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
        double fx, fy, cx, cy;
    };

    class PerpEptsNVector2DSingleFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSBAPointXYZ>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        PerpEptsNVector2DSingleFrame() {}

        virtual void computeError()
        {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);

            // Project 3D endpoints to image coordinates
            Vector2d proj_st_pt = cam_project(st_pt->estimate(), fx, fy, cx, cy);
            Vector2d proj_end_pt = cam_project(end_pt->estimate(), fx, fy, cx, cy);
            // Extract 2D line equation
            Vector2d line_eq = proj_end_pt - proj_st_pt;

            double angle = ComputeAngle2D(_measurement, line_eq);

            _error(0) = angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
        double fx, fy, cx, cy;
    };

    class ParEptsNVector3DMultiFrame : public BaseMultiEdge<3, Vector3d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        ParEptsNVector3DMultiFrame()
        {
            resize(3);
        }

        virtual void computeError()
        {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);
            const VertexSE3Expmap *poseVertex = static_cast<VertexSE3Expmap *>(_vertices[2]);

            // Transform endpoints from world to frame coordinates
            Vector3d st_pt_rot = poseVertex->estimate().map(st_pt->estimate());
            Vector3d end_pt_rot = poseVertex->estimate().map(end_pt->estimate());
            // Extract line eq.
            Vector3d line_eq = end_pt_rot - st_pt_rot;

            double error_angle = ComputeAngle3D(line_eq, _measurement);

            _error(0) = 1 - error_angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class PerpEptsNVector3DMultiFrame : public BaseMultiEdge<3, Vector3d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        PerpEptsNVector3DMultiFrame()
        {
            resize(3);
        }

        virtual void computeError()
        {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);
            const VertexSE3Expmap *poseVertex = static_cast<VertexSE3Expmap *>(_vertices[2]);

            // Transform endpoints from world to frame coordinates
            Vector3d st_pt_rot = poseVertex->estimate().map(st_pt->estimate());
            Vector3d end_pt_rot = poseVertex->estimate().map(end_pt->estimate());

            // Extract line eq.
            Vector3d line_eq = end_pt_rot - st_pt_rot;

            double error_angle = ComputeAngle3D(line_eq, _measurement);

            _error(0) = error_angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class ParEptsNVector2DMultiFrame : public BaseMultiEdge<3, Vector3d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        ParEptsNVector2DMultiFrame()
        {
            resize(3);
        }

        virtual void computeError()
        {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);
            const VertexSE3Expmap *pose_vertex = static_cast<VertexSE3Expmap *>(_vertices[2]);

            // Transform endpoints from world to frame coordinates and project them to image coordinates
            Vector2d proj_st_pt = cam_project(pose_vertex->estimate().map(st_pt->estimate()), fx, fy, cx, cy);
            Vector2d proj_end_pt = cam_project(pose_vertex->estimate().map(end_pt->estimate()), fx, fy, cx, cy);

            // Extract line eq.
            Vector2d line_eq = proj_end_pt - proj_st_pt;

            // Note that the _measurement is in homogeneous coordinates
            double angle = ComputeAngle2D(_measurement, line_eq);

            _error(0) = 1 - angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        double fx, fy, cx, cy;
    };

    class PerpEptsNVector2DMultiFrame : public BaseMultiEdge<3, Vector3d>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        PerpEptsNVector2DMultiFrame()
        {
            resize(3);
        }

        virtual void computeError()
        {
            const g2o::VertexSBAPointXYZ *st_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[0]);
            const g2o::VertexSBAPointXYZ *end_pt = static_cast<g2o::VertexSBAPointXYZ *>(_vertices[1]);
            const VertexSE3Expmap *pose_vertex = static_cast<VertexSE3Expmap *>(_vertices[2]);

            // Transform endpoints from world to frame coordinates and project them to image coordinates
            Vector2d proj_st_pt = cam_project(pose_vertex->estimate().map(st_pt->estimate()), fx, fy, cx, cy);
            Vector2d proj_end_pt = cam_project(pose_vertex->estimate().map(end_pt->estimate()), fx, fy, cx, cy);

            // Extract line eq.
            Vector2d line_eq = proj_end_pt - proj_st_pt;

            // Note that the _measurement is in homogeneous coordinates
            double angle = ComputeAngle2D(_measurement, line_eq);

            _error(0) = angle;
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        double fx, fy, cx, cy;
    };

    class DistPt2Line2DMultiFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        DistPt2Line2DMultiFrame() {}

        virtual void computeError()
        {
            const VertexSE3Expmap *v1 = static_cast<VertexSE3Expmap *>(_vertices[1]);
            const VertexSBAPointXYZ *v2 = static_cast<VertexSBAPointXYZ *>(_vertices[0]);

            Vector3d obs = _measurement;
            Vector2d proj_pt = cam_project(v1->estimate().map(v2->estimate()), fx, fy, cx, cy);

            // Point to line distance in image coordinates
            _error(0) = obs(0) * proj_pt(0) + obs(1) * proj_pt(1) + obs(2);
            _error(1) = 0.0;
            _error(2) = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        double fx, fy, cx, cy;
    };

    class DistPt2Line2DMultiFrameOnlyPose : public BaseUnaryEdge<3, Vector3d, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        DistPt2Line2DMultiFrameOnlyPose() {}

        virtual void computeError()
        {
            const VertexSE3Expmap *v1 = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Vector3d obs_hom_coord = _measurement;
            // Transform endpoint from world to frame, and project it in img coord.
            Vector2d pt_proj = cam_project(v1->estimate().map(Xw), fx, fy, cx, cy);

            // Dist Point to line
            _error(0) = obs_hom_coord(0) * pt_proj(0) + obs_hom_coord(1) * pt_proj(1) + obs_hom_coord(2);
            _error(1) = 0;
            _error(2) = 0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }

        Vector3d Xw; // Non optimizable line endpoint in world coord.
        double fx, fy, cx, cy;
    };

    class Par2Vectors3DMultiFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Par2Vectors3DMultiFrame() {}

        virtual void computeError()
        {
            const VertexSE3Expmap *poseVertex = static_cast<VertexSE3Expmap *>(_vertices[1]);
            const VertexSBAPointXYZ *manhAxisVertex = static_cast<VertexSBAPointXYZ *>(_vertices[0]);

            // Rotate the associated Manh. axis
            Eigen::Vector3d manh_axis_vertex = manhAxisVertex->estimate();
            const g2o::SE3Quat tf = poseVertex->estimate();
            const Eigen::Quaterniond w2n_quat = tf.rotation();
            Eigen::Vector3d  fr_coord_line_manh = w2n_quat * manh_axis_vertex;

            // Manh. axis and 3D line eq. frame angle difference
            _error[0] = 1 - ComputeAngle3D(fr_coord_line_manh, _measurement);
            _error[1] = 0.0;
            _error[2] = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };

    class Perp2Vectors3DMultiFrame : public BaseBinaryEdge<3, Vector3d, g2o::VertexSBAPointXYZ, g2o::VertexSE3Expmap>
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        Perp2Vectors3DMultiFrame() {}

        virtual void computeError()
        {
            const VertexSE3Expmap *poseVertex = static_cast<VertexSE3Expmap *>(_vertices[1]);
            const VertexSBAPointXYZ *manhAxisVertex = static_cast<VertexSBAPointXYZ *>(_vertices[0]);
            // Rotate the associated Manh. axis
            Eigen::Vector3d manh_axis_vertex = manhAxisVertex->estimate();
            const g2o::SE3Quat tf = poseVertex->estimate();
            const Eigen::Quaterniond w2n_quat = tf.rotation();
            Eigen::Vector3d fr_coord_line_manh = w2n_quat * manh_axis_vertex;

            // Manh. axis and 3D line eq. frame angle difference
            _error[0] = ComputeAngle3D(fr_coord_line_manh, _measurement);
            _error[1] = 0.0;
            _error[2] = 0.0;
        }

        bool read(std::istream &is)
        {
            for (int i = 0; i < 3; i++)
            {
                is >> _measurement[i];
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            }
            return true;
        }

        bool write(std::ostream &os) const
        {
            for (int i = 0; i < 3; i++)
            {
                os << measurement()[i] << " ";
            }

            for (int i = 0; i < 3; ++i)
            {
                for (int j = i; j < 3; ++j)
                {
                    os << " " << information()(i, j);
                }
            }
            return os.good();
        }
    };
}

#endif // g2o_MSC_H
