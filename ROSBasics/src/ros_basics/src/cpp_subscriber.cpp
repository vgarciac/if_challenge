#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"

ros::WallTime start_, end_;
float delta_time = 0;

ros::Publisher status_pub;

/**
 * Simple CPP subscriber
 */
void BasicTopicCallback(const std_msgs::String::ConstPtr& msg)
{
    // Update loop rate measure
    end_ = ros::WallTime::now();
    delta_time = (end_ - start_).toSec();
    start_ = ros::WallTime::now();

    // Send real rate, frequency at which messages are received
    std_msgs::Float32 real_rate;
    real_rate.data = 1/delta_time;
    status_pub.publish(real_rate);

    ROS_INFO("CPP_Received: [%s], Frequency: [%0.2f Hz]", msg->data.c_str(), 1/delta_time);
}

int main(int argc, char **argv)
{
    start_ = ros::WallTime::now();

    //Initialize ROS
    ros::init(argc, argv, "cpp_subscriber");

    //Create a handler to process the node 
    ros::NodeHandle node;

    // Create subscriber object
    ros::Subscriber sub = node.subscribe("basic_topic", 10, BasicTopicCallback);

    // Create publisher object 
    status_pub = node.advertise<std_msgs::Float32>("cpp_subscriber_rate", 10);

    // Wait for Callbacks
    ros::spin();

  return 0;
}