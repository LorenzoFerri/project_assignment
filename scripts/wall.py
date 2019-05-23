#!/usr/bin/env python
import rospy
import sys

import numpy as np
from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from math import cos, sin, asin, tan, atan2
# msgs and srv for working with the set_model_service
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from std_srvs.srv import Empty
from enum import Enum
from math import sqrt

# a handy tool to convert orientations
from tf.transformations import euler_from_quaternion, quaternion_from_euler

Mode = Enum('Mode', 'forward adjust turn')


class BasicThymio:

    def __init__(self, thymio_name):
        """init"""
        self.thymio_name = thymio_name
        rospy.init_node('basic_thymio_controller', anonymous=True)
        self.sensors = [10, 10, 10, 10, 10, 10, 10]
        # Publish to the topic '/thymioX/cmd_vel'.
        self.velocity_publisher = rospy.Publisher(self.thymio_name + '/cmd_vel',
                                                  Twist, queue_size=10)

        # A subscriber to the topic '/turtle1/pose'. self.update_pose is called
        # when a message of type Pose is received.
        self.pose_subscriber = rospy.Subscriber(self.thymio_name + '/odom',
                                                Odometry, self.update_state)

        rospy.Subscriber(self.thymio_name + '/proximity/left',
                         Range, lambda data: self.set_sensor(data, 0))
        rospy.Subscriber(self.thymio_name + '/proximity/center_left',
                         Range, lambda data: self.set_sensor(data, 1))
        rospy.Subscriber(self.thymio_name + '/proximity/center',
                         Range, lambda data: self.set_sensor(data, 2))
        rospy.Subscriber(self.thymio_name + '/proximity/center_right',
                         Range, lambda data: self.set_sensor(data, 3))
        rospy.Subscriber(self.thymio_name + '/proximity/right',
                         Range, lambda data: self.set_sensor(data, 4))
        rospy.Subscriber(self.thymio_name + '/proximity/rear_right',
                         Range, lambda data: self.set_sensor(data, 5))
        rospy.Subscriber(self.thymio_name + '/proximity/rear_left',
                         Range, lambda data: self.set_sensor(data, 6))

        self.current_pose = Pose()
        self.current_twist = Twist()
        # publish at this rate
        self.rate = rospy.Rate(10)
        self.mode = Mode.forward
        self.max_range = 0

    def set_sensor(self, data, index):
        self.sensors[index] = data.range
        self.max_range = data.max_range
        if(reduce((lambda x, y: x or y), [x < data.max_range - 0.04 for x in self.sensors])
                and self.mode == Mode.forward):
            self.mode = Mode.adjust

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
                orientation[0], orientation[0], orientation[0], axes='sxyz')
            model_state.pose.orientation.x = qto[0]
            model_state.pose.orientation.y = qto[1]
            model_state.pose.orientation.z = qto[2]
            model_state.pose.orientation.w = qto[3]
            # a Twist can also be set but not recomended to do it in a service
            gms = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
            response = gms(model_state)
            return response
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def update_state(self, data):
        """A new Odometry message has arrived. See Odometry msg definition."""
        # Note: Odmetry message also provides covariance
        self.current_pose = data.pose.pose
        self.current_twist = data.twist.twist
        quat = (
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w)
        (roll, pitch, yaw) = euler_from_quaternion(quat)

    def move_forward(self):
        """Moves the migthy thymio"""

        vel_msg = Twist()
        vel_msg.linear.x = 0.2  # m/s
        vel_msg.angular.z = -0.02  # rad/s\
        while self.mode == Mode.forward:
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
        vel_msg.linear.x = 0
        self.velocity_publisher.publish(vel_msg)
        self.rate.sleep()
        self.adjust()

    def adjust(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0  # m/s
        vel_msg.angular.z = 0  # rad/s\
        print('I reached a wall, positioning my self orthogonal to it.')
        while(abs(self.sensors[1] - self.sensors[3]) > 0.0001):
            vel_msg.angular.z = -(self.sensors[1] - self.sensors[3]) * 6
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
        print('I\'m orthogonal, starting to turn 180 degree, this may take a few seconds')
        print(self.sensors)
        vel_msg.angular.z = 0  # rad/s\
        self.velocity_publisher.publish(vel_msg)
        self.rate.sleep()
        self.turn()

    def turn(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0  # m/s
        vel_msg.angular.z = 0  # rad/s\
        while((self.sensors[5] == self.max_range
               or self.sensors[6] == self.max_range)
              or abs(self.sensors[5] - self.sensors[6]) > 0.00001):
            vel_msg.angular.z = (-(self.sensors[5] -
                                   self.sensors[6]) * 7) or 0.6
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
        print('Turned 180 degree, now walking forward for 2 meters')
        vel_msg.angular.z = 0  # rad/s\
        self.velocity_publisher.publish(vel_msg)
        self.rate.sleep()
        self.forward_two_meters()

    def forward_two_meters(self):
        vel_msg = Twist()
        vel_msg.linear.x = 0.2  # m/s
        vel_msg.angular.z = 0  # rad/s\
        starting_x = self.current_pose.position.x
        starting_y = self.current_pose.position.y
        while(sqrt((starting_x - self.current_pose.position.x)**2+(starting_y - self.current_pose.position.y)**2)-2 < 0):
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
        vel_msg.linear.x = 0
        self.velocity_publisher.publish(vel_msg)
        self.rate.sleep()
        rospy.spin()


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
    # Teleport the robot to a certain pose. If pose is different to the
    # origin of the world, you must account for a transformation between
    # odom and gazebo world frames.
    # NOTE: The goal of this step is *only* to show the available
    # tools. The launch file process should take care of initializing
    # the simulation and spawning the respective models

    # thymio.thymio_state_service_request([0print.,0.,0.], [0.,0.,0.])
    # rospy.sleep(1.)
    thymio.move_forward()
