#include "ros/ros.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32.h"
#include <dynamic_reconfigure/server.h>
#include <ros_basics/RateConfig.h>



#include <sstream>
const long int big_number = 1000000;

ros::WallTime start_, end_;
float delta_time = 0.1;
int dynamic_param = 10;

void foo()
{
    for(int i = 0 ; i < big_number ; i++){}
    return;
}

void callback(ros_basics::RateConfig &config, uint32_t level) {
    // Save dynamic parameter
    dynamic_param = config.publisher_rate * 10;
}

/**
 * Simple CPP publisher
 */
int main(int argc, char **argv)
{

    //Initialize ROS
    ros::init(argc, argv, "cpp_publisher");
    
    //Create a handler to process the node 
    ros::NodeHandle node;

    dynamic_reconfigure::Server<ros_basics::RateConfig> server;
    dynamic_reconfigure::Server<ros_basics::RateConfig>::CallbackType f;

    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);

    // Create publisher object
    ros::Publisher basic_topic_pub = node.advertise<std_msgs::String>("basic_topic", 10);

    // Create publisher object 
    ros::Publisher status_pub = node.advertise<std_msgs::Float32>("cpp_publisher_rate", 10);
    ros::Publisher required_value_pub = node.advertise<std_msgs::Float32>("cpp_required_rate", 10);

    ros::Rate loop_rate(dynamic_param);

    // A count of how many messages we have sent.
    int count = 0;
    start_ = ros::WallTime::now();

    while (ros::ok())
    {
        // Set new loop rate
        ros::Rate loop_rate(dynamic_param);

        // Create message objects
        std_msgs::String msg;
        std_msgs::Float32 real_rate, required_rate;

        std::stringstream ss;
        ss << "ifollow Challenge - Victor" << count;
        msg.data = ss.str();

        // Publish message
        basic_topic_pub.publish(msg);
        real_rate.data = 1/delta_time;
        status_pub.publish(real_rate);
        required_rate.data = dynamic_param;
        required_value_pub.publish(required_rate);

        // Update loop rate measure
        end_ = ros::WallTime::now();
        delta_time = (end_ - start_).toSec();
        start_ = ros::WallTime::now();

        ROS_INFO("CPP_Sending: [%s], Frequency: [%0.2f Hz]", msg.data.c_str(), 1/delta_time);

        // Check parameter config callback
        ros::spinOnce();

        ++count;
        loop_rate.sleep();
    }

    return 0;
}