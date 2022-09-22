#!/usr/bin/env python
import rospy
import numpy as np
from geometry_msgs.msg import WrenchStamped

# force and torque sensor

class FTSensor:
    def __init__(self, ft_topic):
        self.topic = ft_topic
        self.ft_sub = rospy.Subscriber(self.topic, WrenchStamped, self._ft_cb)
        self.record = []
        self.number_of_points = 8

    def _ft_cb(self,data):
        force = data.wrench.force
        if len(self.record) <= self.number_of_points:
            self.record.append([force.x,force.y,force.z])
        else:
            self.record.pop(0)
            self.record.append([force.x,force.y,force.z])
        #print(self.record)

    def _moving_average(self):
        force_record = np.array(self.record)
        return np.mean(force_record,axis=0)

    def forces(self):
        data = self._moving_average()
        return data

if __name__ == '__main__':
    rospy.init_node("ft_smoother", anonymous=True, log_level=rospy.INFO)
    rate = rospy.Rate(1)
    sensor = FTSensor('ft_sidebar')
    try:
        while not rospy.is_shutdown():
            print("smoothedw with average moving",sensor.forces())
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
