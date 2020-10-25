#!/usr/bin/env python
import rospy
from std_msgs.msg import *

end_ = None
start_ = None

class Python_subs:
    def __init__(self):

        # Init node
        rospy.init_node('py_subscriber', anonymous=True)
                
        # Create subscriber
        rospy.Subscriber("basic_topic", String, self.callback)

        # Create publisher
        self.status_pub = rospy.Publisher('py_subscriber_rate', Float32, queue_size=10)

        # Get init time
        self.start_ = rospy.get_time()

    def callback(self, data):
                    
        # Update loop rate measure
        self.end_ = rospy.get_time()
        delta_time = (self.end_ - self.start_)
        self.start_ = rospy.get_time()

        # Publish String and publishing loop rate
        self.status_pub.publish(1/delta_time)
        
        # Report info in console
        rospy.loginfo("PY_Received: [%s], Frequency: [%0.2f Hz]" % (data.data, 1/delta_time))


if __name__ == '__main__':

    python_subscriber = Python_subs()
    if not rospy.is_shutdown():
        rospy.spin()