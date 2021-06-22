
/* ROS Headers */
#include<ros/ros.h>
#include<std_msgs/String.h>
#include<sensor_msgs/PointCloud2.h>
#include<ros/package.h>
#include<pcl_ros/point_cloud.h>

/* I/O */
#include<pcl/point_types.h>
#include<pcl/io/pcd_io.h>

/* Features */
#include <pcl/features/normal_3d.h>

/* Filters */
#include<pcl/filters/crop_box.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/filter_indices.h> 
#include <pcl/filters/statistical_outlier_removal.h>

/* Visualization */
#include <pcl/visualization/pcl_visualizer.h>

/* Timer */
#include <pcl/common/time.h>

typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointCloudT;

class ObjectSegmentation
{ 
public:

    //Camera members
    bool pointcloud_done_cam1_ = false;
    ros::Subscriber cam1_pointcloud_sub_;
    std::string frame_id_ = "";
    std::string camera_topics_;

    ros::Publisher pcd_cloud_pub_;
    ros::NodeHandle nh_;
    
    PointCloudT::Ptr box_filtered_cloud_ptr;
    PointCloudT::Ptr outlier_removal_output_cloud_ptr;
    PointCloudT::Ptr cam1_scene_cloud_ptr_;
    PointCloudT::Ptr cam1_scene_cloud_ptr_filtered_;

    pcl::search::Search<pcl::PointXYZRGBNormal>::Ptr tree;

    pcl::PointCloud<pcl::Normal>::Ptr normals;

    ObjectSegmentation();

    // ROS Callback function
    void pointcloudInfoCb(const sensor_msgs::PointCloud2& scene_cloud);

    // Load Input PCD
    void loadInputPCD();

    // Process the point cloud
    void processPointCloud();

    // Apply Box Filter
    void applyBoxFilter();

    // Remove parallel planes
    void applyPlaneSegmentation();

    // Remove Stastical outliers
    void applyStatisticalOutlierRemovalFilter();

    // Apply Region Growing CLustering
    void applyRegionGrowingSegmentation();

    // Display processed pcd on rviz
    void publishRGBCloudToRviz(pcl::PointCloud<pcl::PointXYZRGB>::Ptr obj_detection_cloud, ros::Publisher& pub, std::string frame_id);

    // Visualise Point CLoud
    void visualiseProcessed3DImage();

};