#!/usr/bin/env python
import rospy
import sys

import numpy as np
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from math import cos, sin, asin, tan, atan2
# msgs and srv for working with the set_model_service
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from matplotlib import pyplot as plt
from std_srvs.srv import Empty
from dynamic_reconfigure.msg import Config
from dynamic_reconfigure.parameter_generator import *
import cv2
import time

# a handy tool to convert orientations
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class BasicThymio:

    def __init__(self, thymio_name):
        """init"""
        self.thymio_name = thymio_name
        rospy.init_node('basic_thymio_controller', anonymous=True)
        time.sleep(2)
        self.velocity_publisher = rospy.Publisher(self.thymio_name + '/cmd_vel',
                                                  Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber(self.thymio_name + '/odom',
                                                Odometry, self.update_state)
        self.camera_subscriber = rospy.Subscriber(self.thymio_name + '/camera/image_raw',
                                                  Image, self.update_image, queue_size=1)

        self.camera_pitch_pub = rospy.Publisher(
            self.thymio_name + '/camera_pitch_controller/parameter_updates', Config, queue_size=1)
        self.current_pose = Pose()
        self.current_twist = Twist()
        self.rate = rospy.Rate(10)
        self.pixels = None

    def thymio_state_service_request(self, position, orientation):
        """Request the service (set thymio state values) exposed by
        the simulated thymio. A teleportation tool, by default in gazebo world frame.
        Be aware, this does not mean a reset (e.g. odometry values)."""
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            model_state = ModelState()
            model_state.model_name = self.thymio_name
            model_state.reference_frame = ''  # the frame for the pose information
            model_state.pose.position.x = position[0]
            model_state.pose.position.y = position[1]
            model_state.pose.position.z = position[2]
            qto = quaternion_from_euler(
                orientation[0], orientation[1], orientation[2], axes='sxyz')
            model_state.pose.orientation.x = qto[0]
            model_state.pose.orientation.y = qto[1]
            model_state.pose.orientation.z = qto[2]
            model_state.pose.orientation.w = qto[3]
            gms = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            response = gms(model_state)
            return response
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def update_state(self, data):
        """A new Odometry message has arrived. See Odometry msg definition."""
        self.current_pose = data.pose.pose
        self.current_twist = data.twist.twist

    def update_image(self, img):
        """A new Odometry message has arrived. See Odometry msg definition."""
        self.pixels = np.fromstring(img.data, dtype=np.dtype(
            np.uint8)).reshape(480, 640, 3)

    def sensor_discovery(self):
        for _topic, _type in rospy.get_published_topics():
            print _topic, _type
        # while(np.array_equal(current_pixels, self.pixels)):
        #     print('a')


def usage():
    return "Wrong number of parameters. basic_move.py [thymio_name]"


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        thymio_name = sys.argv[1]
        print "Now working with robot: %s" % thymio_name
    else:
        print usage()
        sys.exit(1)
    thymio = BasicThymio(thymio_name)

    # thymio.thymio_state_service_request([3., 3., 0.], [0, 1, 0, 0])
    save_flag = False
    start = False
    distance = 0.25
    angle = -np.pi/3
    while distance <= 1.5:
        if thymio.pixels is not None:
            cv2.imshow("Image", thymio.pixels)
            if save_flag:
                cv2.imwrite(
                    '../dataset/image_'+str(distance)+'_'+str(angle)+'.png', thymio.pixels)
                save_flag = False
                start = True
                angle += 0.02
                if angle > np.pi/3:
                    angle = -np.pi/3
                    distance += 0.02
                    print distance
            thymio.pixels = None

        p = cv2.waitKey(1)

        if p == 27:  # esc to quit
            thymio.sensor_discovery()

        if p == ord(' '):
            start = True

        # if p == ord('a'):

        if start:
            start = False
            thymio.thymio_state_service_request(
                [0., distance, 0.], [0, 0, (-np.pi/2)+angle])
            thymio.pixels = None
            save_flag = True

        thymio.rate.sleep()
