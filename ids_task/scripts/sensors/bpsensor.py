import numpy as np
from gazebo_msgs.msg import ContactsState
import rospy

class BumpSensor:
    def __init__(self, topic='/bumper_plug'):
        self.topic = topic
        self.contact_sub = rospy.Subscriber(self.topic, ContactsState, self._contact_cb)
        self.touched = False

    def connected(self):
        return self.touched

    def _contact_cb(self, data):
        states = data.states
        if len(states) > 0:
            self.touched = True
        else:
            self.touched = False
        # print(self.touched)

    def check_sensor_ready(self):
        self.touched = False
        rospy.logdebug("Waiting for /bumper_plug to be READY...")
        while not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/bumper_plug", ContactsState, timeout=5.0)
                self.touched = len(data.states) > 0
                rospy.logdebug("Current /bumper_plugs READY=>")
            except:
                rospy.logerr("Current /bumper_plug not ready yet, retrying for getting  /bumper_plug")
