#ifndef LIDAR_PROCESS_H
#define LIDAR_PROCESS_H

#include <cmath>
#include <vector>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/pca.h>
#include <pcl/common/centroid.h>
#include <pcl/common/distances.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <set>
#include <pcl/io/pcd_io.h>
#include <boost/format.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/console/time.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <visualization_msgs/MarkerArray.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>
#include <jsk_recognition_msgs/BoundingBox.h>
#include <Eigen/Dense>
#include <std_msgs/Float64MultiArray.h>

//define
#define Point2 pcl::PointXYZI
using namespace std;

typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;

extern pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
extern pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_grid_cloud;
extern pcl::ModelCoefficients::Ptr plane_coefficients;
extern pcl::PointIndices::Ptr plane_inliers;
extern pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud;
extern pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud;
extern pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;

class LidarProcess{
    protected:  
        ros::NodeHandle nh;

        ros::Publisher pub_voxel;
        ros::Publisher pub_roi;
        ros::Publisher pub_ransac_ground;
        ros::Publisher pub_ransac_obstacle;
        ros::Publisher pub_cluster;
        ros::Publisher marker_pub;

        ros::Subscriber point_cloud_sub;

    public:
        LidarProcess();
        ~LidarProcess();

        void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& scan);

        void setRoi(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);

        void voxelGridFilter(sensor_msgs::PointCloud2& filtered_cloud_msg, pcl::PointCloud<pcl::PointXYZ>::Ptr& voxel_grid_cloud);

        void ranSac(pcl::PointCloud<pcl::PointXYZ>::Ptr& voxel_grid_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud, pcl::ModelCoefficients::Ptr plane_coefficients, pcl::PointIndices::Ptr plane_inliers);

        void euclideanCluster(pcl::PointCloud<pcl::PointXYZ>::Ptr& obstacle_cloud);

        void publishLshapeFittingBox(const std::vector<std::vector<float>>& cluster_centers, const std::vector<std::vector<float>>& cluster_length, const std::vector<std::vector<float>>& cluster_quat);

        sensor_msgs::PointCloud2 filtered_cloud_msg;

        sensor_msgs::PointCloud2 ground_cloud_msg;


        
};

#endif