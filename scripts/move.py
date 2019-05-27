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
import cv2
import time
import torch
from torchvision import transforms

from Model import Net

# a handy tool to convert orientations
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class BasicThymio:

    def __init__(self, thymio_name):
        """init"""
        self.net = Net()
        self.net.load_state_dict(torch.load('./cnn', 'cpu'))
        self.net.eval()
        self.thymio_name = thymio_name
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
        rospy.init_node('basic_thymio_controller', anonymous=True)
        time.sleep(5)

        self.velocity_publisher = rospy.Publisher(self.thymio_name + '/cmd_vel',
                                                  Twist, queue_size=10)
        self.pose_subscriber = rospy.Subscriber(self.thymio_name + '/odom',
                                                Odometry, self.update_state)

        self.camera_subscriber = rospy.Subscriber(self.thymio_name + '/camera/image_raw',
                                                  Image, self.update_image, queue_size=1)

        self.current_pose = Pose()
        self.current_twist = Twist()
        self.rate = rospy.Rate(10)

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

    def update_image(self, img):
        """A new Odometry message has arrived. See Odometry msg definition."""
        pixels = np.fromstring(img.data, dtype=np.dtype(
            np.uint8)).reshape(480, 640, 3)
        pts1 = np.float32([[280, 0], [360, 0], [0, 480], [640, 480]])
        pts2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(pixels, matrix, (640, 480))
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(image.shape)
        # image = image.transpose((2, 0, 1))
        image = self.transform(image)
        image = torch.unsqueeze(image, 0)
        res = self.net(image)[0]
        dist = res[0].item()/10
        angle = res[1].item()/10
        overlay = cv2.putText(pixels,
                              str(-(self.current_pose.position.x-0.25)),
                              (10, 420),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (0, 0, 0))
        overlay = cv2.putText(overlay,
                              str(self.current_pose.orientation.z),
                              (400, 420),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (0, 0, 0))
        overlay = cv2.putText(overlay,
                              str(dist),
                              (10, 440),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (150, 0, 0))
        overlay = cv2.putText(overlay,
                              str(angle),
                              (400, 440),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5,
                              (150, 0, 0))
        cv2.imshow("Image", overlay)
        p = cv2.waitKey(1)
        if p == ord('w'):
            vel_msg = Twist()
            vel_msg.linear.x = 0.2
            vel_msg.angular.z = self.current_twist.angular.z
            self.velocity_publisher.publish(vel_msg)
        if p == ord('a'):
            vel_msg = Twist()
            vel_msg.linear.x = self.current_twist.linear.x
            vel_msg.angular.z = 0.2
            self.velocity_publisher.publish(vel_msg)
        if p == ord('s'):
            vel_msg = Twist()
            vel_msg.linear.x = 0
            vel_msg.angular.z = self.current_twist.angular.z
            self.velocity_publisher.publish(vel_msg)
        if p == ord('x'):
            vel_msg = Twist()
            vel_msg.linear.x = -0.2
            vel_msg.angular.z = self.current_twist.angular.z
            self.velocity_publisher.publish(vel_msg)
        if p == ord('d'):
            vel_msg = Twist()
            vel_msg.linear.x = self.current_twist.linear.x
            vel_msg.angular.z = -0.2
            self.velocity_publisher.publish(vel_msg)
        if p == ord('e'):
            vel_msg = Twist()
            vel_msg.linear.x = self.current_twist.linear.x
            vel_msg.angular.z = 0
            self.velocity_publisher.publish(vel_msg)
        if p == ord('t'):
            self.thymio_state_service_request(
                [0., 0.25, 0.], [0, 0, -(np.pi/2)])


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

    rospy.spin()
