#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/conversions.h>
#include <pcl_ros/point_cloud.h>
#include <nav_msgs/OccupancyGrid.h>
#include "map/map.hpp"

using namespace std;
using PointCloud = pcl::PointCloud<pcl::PointXYZ>;

void processPointCloud(const PointCloud::Ptr& cloud, GridMap& gridmap) {
    // Downsample for efficiency
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.05f, 0.05f, 0.05f);
    PointCloud::Ptr ds_cloud(new PointCloud);
    vg.filter(*ds_cloud);

    // Ground plane removal
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.05);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    seg.setInputCloud(ds_cloud);
    seg.segment(*inliers, *coefficients);
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(ds_cloud);
    extract.setIndices(inliers);
    extract.setNegative(true);
    PointCloud::Ptr obstacles(new PointCloud);
    extract.filter(*obstacles);

    // Segmentation into obstacles
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(obstacles);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.3);
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(obstacles);
    vector<pcl::PointIndices> cluster_indices;
    ec.extract(cluster_indices);

    for (const auto& cluster : cluster_indices) {
        pcl::PointCloud<pcl::PointXYZ> cluster_cloud;
        for (int i : cluster.indices) {
            cluster_cloud.push_back(obstacles->at(i));
        }

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(cluster_cloud, centroid);
        double max_radius = 0.0;
        double min_z = numeric_limits<double>::max();
        double max_z = numeric_limits<double>::lowest();

        for (const auto& pt : cluster_cloud) {
            double dx = pt.x - centroid[0];
            double dy = pt.y - centroid[1];
            double r = sqrt(dx*dx + dy*dy);
            max_radius = max(max_radius, r);
            min_z = min(min_z, pt.z);
            max_z = max(max_z, pt.z);
        }

        double height = max_z - min_z;
        ObsCylinder cyl(centroid.block<3,1>(0,0), height, max_radius);
        gridmap.push_obs(cyl);
    }
}

void publishGridMap(const GridMap& gridmap, ros::Publisher& grid_map_pub) {
    nav_msgs::OccupancyGrid grid_msg;
    grid_msg.header.stamp = ros::Time::now();
    grid_msg.info.resolution = gridmap.resolution();
    grid_msg.info.width = gridmap.size()(0) / gridmap.resolution();
    grid_msg.info.height = gridmap.size()(1) / gridmap.resolution();
    grid_msg.data.resize(grid_msg.info.width * grid_msg.info.height);

    // Populate grid_msg.data from gridmap
    for (size_t i = 0; i < grid_msg.data.size(); ++i) {
        grid_msg.data[i] = gridmap.get_cell(i).occupied ? 100 : 0;
    }

    grid_map_pub.publish(grid_msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "depth_to_map");
    ros::NodeHandle nh;

    // Parameters
    string input_topic = "/camera/depth/points";
    double resolution = 0.05;
    Vector3d map_size(50.0, 10.0, 3.0);

    GridMap gridmap(resolution, map_size);
    ros::Subscriber sub = nh.subscribe<PointCloud>(input_topic, 1, [&](const PointCloud::ConstPtr& msg) {
        processPointCloud(msg, gridmap);
    });

    ros::Publisher grid_map_pub = nh.advertise<nav_msgs::OccupancyGrid>("grid_map", 1);

    ros::Rate rate(10); // 10 Hz
    while (ros::ok()) {
        publishGridMap(gridmap, grid_map_pub);
        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}