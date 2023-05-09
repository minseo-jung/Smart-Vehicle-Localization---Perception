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

ros::Publisher pub_voxel;
ros::Publisher pub_roi;
ros::Publisher pub_ransac_ground;
ros::Publisher pub_ransac_obstacle;
ros::Publisher pub_cluster;
ros::Publisher marker_pub;

void publishMarkers(const std::vector<std::vector<float>>& cluster_centers, const std::vector<std::vector<float>>& cluster_length, const std::vector<std::vector<float>>& cluster_quat);

void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& scan)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*scan, *cloud);

    //3. 차량 기준 사각범위만 받기
    
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(-10.0, 30.0);
    pass.filter(*cloud);

    pass.setInputCloud(cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(-10.0, 10.0);
    pass.filter(*cloud);

    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(-2.0, 1);
    pass.filter(*cloud);

    sensor_msgs::PointCloud2 filtered_cloud_msg;
    pcl::toROSMsg(*cloud, filtered_cloud_msg);
    filtered_cloud_msg.header = scan->header;
    pub_roi.publish(filtered_cloud_msg);


    //voxel

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(filtered_cloud_msg, *filtered_cloud);

    pcl::PointCloud<pcl::PointXYZ> cloudClusterIn;
    pcl::copyPointCloud(*filtered_cloud, cloudClusterIn);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered_v2 (new pcl::PointCloud<pcl::PointXYZ>);
    vg.setInputCloud (cloudClusterIn.makeShared());
    vg.setLeafSize (0.1, 0.1, 0.1);		//복셀 다운샘플링의 단위 (한 복셀의 크기)
    vg.filter (*cloud_filtered_v2);			
  
    pcl::PCLPointCloud2 cloud_v; 			//복셀화된 클라우드 데이터 구조 선언
    sensor_msgs::PointCloud2 output_v; 		//출력할 방식인 PC2 선정 및 이름 output_v 정의
    pcl::toPCLPointCloud2(*cloud_filtered_v2, cloud_v);
    pcl_conversions::fromPCL(cloud_v, output_v);
    output_v.header.frame_id = "rslidar";
    pub_voxel.publish(output_v);

    pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> sac_segmentation;
    sac_segmentation.setOptimizeCoefficients(true);
    sac_segmentation.setModelType(pcl::SACMODEL_PLANE);
    sac_segmentation.setMethodType(pcl::SAC_RANSAC);
    sac_segmentation.setMaxIterations(1000);
    sac_segmentation.setDistanceThreshold(0.1);
    sac_segmentation.setInputCloud(cloud_filtered_v2);
    sac_segmentation.segment(*plane_inliers, *plane_coefficients);

    // Extract plane inliers and transform them to the XY plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr obstacle_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ExtractIndices<pcl::PointXYZ> extract_indices;
    extract_indices.setInputCloud(cloud_filtered_v2);
    extract_indices.setIndices(plane_inliers);
    extract_indices.setNegative(true);
    extract_indices.filter(*obstacle_cloud);
    extract_indices.setNegative(false);
    extract_indices.filter(*plane_cloud);

    sensor_msgs::PointCloud2 ground_cloud_msg;
    pcl::toROSMsg(*plane_cloud, ground_cloud_msg);
    ground_cloud_msg.header.frame_id = "rslidar";
    pub_ransac_ground.publish(ground_cloud_msg);

    sensor_msgs::PointCloud2 obstacle_cloud_msg;
    pcl::toROSMsg(*obstacle_cloud, obstacle_cloud_msg);
    obstacle_cloud_msg.header.frame_id = "rslidar";
    pub_ransac_obstacle.publish(obstacle_cloud_msg);

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud (obstacle_cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (0.3); 			// 0.1cm의 포인트와 포인트 간의 간격
    ec.setMinClusterSize (20);				// 한 군집의 최소 포인트 개수
    ec.setMaxClusterSize (25000);				// 한 군집의 최대 포인트 개수
    ec.setSearchMethod (tree);				// 검색 방법 : tree 
    ec.setInputCloud (obstacle_cloud);	// cloud_filtered_v2에 클러스터링 결과를 입력
    ec.extract (cluster_indices);

    int cluster_id = 0;

    //cluster center
    std::vector<std::vector<float>> cluster_center(cluster_indices.size(),vector<float>(3,0));
    //max x,y,z min x,y,z
    std::vector<std::vector<float>> cluster_length(cluster_indices.size(),vector<float>(3,0));
    //yaw
    std::vector<std::vector<float>> cluster_quat(cluster_indices.size(),vector<float>(4,0));

    for (const auto& indices : cluster_indices)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        extract.setInputCloud(obstacle_cloud);
        extract.setIndices(boost::make_shared<pcl::PointIndices>(indices));
        extract.setNegative(false);
        extract.filter(*cluster_cloud);
        
        pcl::PointCloud<pcl::PointXYZ> projection_cloud;
        pcl::copyPointCloud(*cluster_cloud, projection_cloud);

        Eigen::Vector4f centroid;
        pcl::PointXYZ min_pt, max_pt;

        pcl::compute3DCentroid(projection_cloud, centroid);
        pcl::getMinMax3D(projection_cloud, min_pt, max_pt);

        cluster_center[cluster_id][0] = centroid[0];
        cluster_center[cluster_id][1] = centroid[1];
        cluster_center[cluster_id][2] = centroid[2];

        cluster_length[cluster_id][2] = max_pt.z - min_pt.z;

        for(size_t i = 0; i<projection_cloud.size(); i++){
            projection_cloud.points[i].z = 0;
        }

        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(projection_cloud.makeShared());
        
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();

        Eigen::Vector3f major_axis = eigen_vectors.col(0);

        float angle = acos(major_axis.dot(Eigen::Vector3f::UnitX()));

        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.rotate(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));

        // Apply rotation to point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr rotated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(projection_cloud, *rotated_cloud, transform);

        pcl::getMinMax3D(*rotated_cloud, min_pt, max_pt);

        cluster_length[cluster_id][0] = max_pt.x - min_pt.x;
        cluster_length[cluster_id][1] = max_pt.y - min_pt.y;

        Eigen::Quaternionf rotation_quat(transform.inverse().rotation());

        cluster_quat[cluster_id][0] = rotation_quat.x();
        cluster_quat[cluster_id][1] = rotation_quat.y();
        cluster_quat[cluster_id][2] = rotation_quat.z();
        cluster_quat[cluster_id][3] = rotation_quat.w();

        cluster_id++;
    }

    publishMarkers(cluster_center, cluster_length, cluster_quat);

    pcl::PointCloud<pcl::PointXYZI> TotalCloud; 
    int j = 0;
    double object_center_x,object_center_y,object_center_z;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
    	for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    	{
        	pcl::PointXYZ pt = obstacle_cloud->points[*pit];
          	pcl::PointXYZI pt2;
          	pt2.x = pt.x, pt2.y = pt.y, pt2.z = pt.z;
          	pt2.intensity = (float)(j + 1);
          	TotalCloud.push_back(pt2);
    	}
    j++;
    }
    
    pcl::PCLPointCloud2 cloud_clustered;
    pcl::toPCLPointCloud2(TotalCloud, cloud_clustered);
    sensor_msgs::PointCloud2 output_clustered; 
    pcl_conversions::fromPCL(cloud_clustered, output_clustered);
    output_clustered.header.frame_id = "rslidar";
    pub_cluster.publish(output_clustered); 
}

void publishMarkers(const std::vector<std::vector<float>>& cluster_center, const std::vector<std::vector<float>>& cluster_length, const std::vector<std::vector<float>>& cluster_quat)
{
    // create marker array message
    jsk_recognition_msgs::BoundingBoxArray marker_array;

    marker_array.header.frame_id = "rslidar";
    marker_array.header.stamp = ros::Time::now();

    marker_array.boxes.resize(cluster_length.size());

    // loop through each cluster
    for (int i = 0; i < cluster_length.size(); i++) {

        // create marker message
        jsk_recognition_msgs::BoundingBox marker;
        marker.header.frame_id = "rslidar"; // change "base_link" to your frame ID
        marker.header.stamp = ros::Time::now();
        marker.pose.position.x = cluster_center[i][0];
        marker.pose.position.y = cluster_center[i][1];
        marker.pose.position.z = cluster_center[i][2];
        marker.pose.orientation.x = cluster_quat[i][0];
        marker.pose.orientation.y = cluster_quat[i][1];
        marker.pose.orientation.z = cluster_quat[i][2];
        marker.pose.orientation.w = cluster_quat[i][3];
        marker.dimensions.x = cluster_length[i][0];
        marker.dimensions.y = cluster_length[i][1];
        marker.dimensions.z = cluster_length[i][2];

        // add marker to array
        marker_array.boxes[i] = marker;
    }

    // publish marker array
    marker_pub.publish(marker_array);
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "rslidar_cluster");
	ros::NodeHandle nh;
	ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2> ("/rslidar_points", 100, lidarCallback); //front ouster
    pub_voxel = nh.advertise<sensor_msgs::PointCloud2> ("/rslidar_points_voxelized", 100);
    pub_roi = nh.advertise<sensor_msgs::PointCloud2> ("/rslidar_points_ROI", 100);
    pub_ransac_ground = nh.advertise<sensor_msgs::PointCloud2> ("/rslidar_points_ransac_ground", 100);
    pub_ransac_obstacle = nh.advertise<sensor_msgs::PointCloud2> ("/rslidar_points_ransac_obstacle", 100);
    pub_cluster = nh.advertise<sensor_msgs::PointCloud2> ("/rslidar_points_cluster", 100);
    marker_pub = nh.advertise<jsk_recognition_msgs::BoundingBoxArray>("l_shape_marker", 10);
    
	ros::spin();
}