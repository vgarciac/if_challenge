#!/usr/bin/env python

import rospy
from std_msgs.msg import *
from ros_basics.cfg import RateConfig
from dynamic_reconfigure.server import Server

class Python_pub:
	
    def __init__(self):

        # Init node
        rospy.init_node('py_publisher', anonymous=True)
        # Create publisher
        self.basic_topic_pub = rospy.Publisher('basic_topic', String, queue_size=10)
        self.status_pub = rospy.Publisher('py_publisher_rate', Float32, queue_size=10)
        self.required_value_pub = rospy.Publisher('py_required_rate', Float32, queue_size=10)
        

        # Define configurable parameter
        self.param = {'publisher_rate': 10}

        # Defining publish rate
        self.rate = rospy.Rate(10)

        # Connect to dynamic parameter server
        srv = Server(RateConfig, self.reconfig)

        # Get init time
        self.start_ = rospy.get_time()
        self.delta_time = 0.1

    def py_publisher(self):

        while not rospy.is_shutdown():
            # Read dynamic parameters
            dynamic_param = self.param['publisher_rate'] * 10

            # Set new publishing rate
            self.rate = rospy.Rate(dynamic_param)
            

            # Create String message
            message = "ifollow Challenge - Victor"

            # Publish String and publishing loop rate
            self.basic_topic_pub.publish(message)
            self.status_pub.publish(1/self.delta_time)
            self.required_value_pub.publish(dynamic_param)
            
            # Update loop rate measure
            self.end_ = rospy.get_time()
            self.delta_time = (self.end_ - self.start_)
            self.start_ = rospy.get_time()

            # Report info in console
            rospy.loginfo("PY_Sending: [%s], Frequency: [%f Hz]" % (message, 1/self.delta_time))
            
            # Sleep
            self.rate.sleep()

    def reconfig(self, config, level):
        # Save parameter recived from server
        self.param = config
        return config

if __name__ == '__main__':
    python_publisher = Python_pub()
    try:
        # Run publishing function until Ctrl-C is called
        python_publisher.py_publisher()
    except rospy.ROSInterruptException:
        pass
