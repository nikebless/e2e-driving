#!/usr/bin/env python
import sys
import os

import message_filters
import numpy as np
import pandas as pd
import rospy
from pacmod_msgs.msg import SystemRptInt
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
from autoware_msgs.msg import VehicleCmd, Gear
from cv_bridge import CvBridge
import cv2

from sklearn.neighbors import BallTree

from tf.transformations import euler_from_quaternion

print('PID:', os.getpid())


class VelocityModel:
    def __init__(self, positions_parquet='positions.parquet', vector_velocity=30):
        self.vector_velocity = vector_velocity
        self.positions_df = pd.read_parquet(positions_parquet)
        self.tree = BallTree(self.positions_df[["position_x", "position_y", "position_x2", "position_y2"]])

    def find_speed_for_position(self, x, y, yaw):
        x2 = x + (self.vector_velocity * np.cos(yaw))
        y2 = y + (self.vector_velocity * np.sin(yaw))

        closest = self.tree.query([[x, y, x2, y2]])
        distance = closest[0][0][0]
        index = closest[1][0][0]
        return self.positions_df.iloc[index]["vehicle_speed"], distance

class NvidiaE2E:
    def __init__(self):
        self.steering_ratio = rospy.get_param('~steering_ratio')
        self.convert_to_rgb = rospy.get_param('~convert_to_rgb')

        self.speed = 0.0
        speed_parquet_path = rospy.get_param('~speed_parquet_path')
        self.max_closest_points_distance = rospy.get_param('~max_closest_point_distance')
        print("Initializing speed model from ", speed_parquet_path)
        self.speed_model = VelocityModel(positions_parquet=speed_parquet_path)

        current_pose_sub = message_filters.Subscriber('/current_pose', PoseStamped)
        current_pose_sub.registerCallback(self.pose_callback)

        turn_signal_sub = message_filters.Subscriber('/pacmod/parsed_tx/turn_rpt', SystemRptInt)
        turn_signal_sub.registerCallback(self.turn_signal_callback)
        self.turn_signal = SystemRptInt.TURN_NONE

        self.xmin = rospy.get_param('~crop_xmin')
        self.xmax = rospy.get_param('~crop_xmax')
        self.ymin = 520
        self.ymax = 864
        self.scale = 0.2

        height = self.ymax - self.ymin
        width = self.xmax - self.xmin
        self.scaled_width = int(self.scale * width)
        self.scaled_height = int(self.scale * height)

        front_wide_camera_sub = message_filters.Subscriber("/interfacea/link2/image/compressed", CompressedImage, queue_size=1, buff_size=2 ** 32)
        front_wide_camera_sub.registerCallback(self.callback)

        self.pub_img = rospy.Publisher('nvidia_wide_center/cropped_image', Image, queue_size=1)

        self.bridge = CvBridge()
        self.img = None
        self.comb_img = None
        self.prev_img = None
        self.stamp = None

        # Publishers
        self.pub_cmd = rospy.Publisher('/vehicle_cmd', VehicleCmd, queue_size=1)
        self.pub_img = rospy.Publisher('/cropped_image', Image, queue_size=1)

        self.speed_adj_multiple = rospy.get_param('~speed_adj_multiple')
        print("Adjusting speed using multiple: ", self.speed_adj_multiple)
        self.max_speed = rospy.get_param('~max_speed')
        print("Max speed limit: ", self.max_speed)
        self.min_speed = rospy.get_param('~min_speed')
        print("Min speed limit: ", self.min_speed)

        self.model_type = rospy.get_param('~model_type')
        print("Model type: ", self.model_type)

        inference_engine = rospy.get_param('~inference_engine')
        steering_onnx_file_path = rospy.get_param('~steering_file_path')
        if inference_engine == 'ort':
            from onnx_model import OnnxModel

            self.steering_model = OnnxModel(steering_onnx_file_path)
            print(self.steering_model.session.get_inputs()[0].shape)
        elif inference_engine == 'trt':
            from tensorrt_model import TensorrtModel

            self.steering_model = TensorrtModel(steering_onnx_file_path)

    def publish_img(self, img):
        msg = self.bridge.cv2_to_imgmsg(img, encoding='rgb8')
        msg.header.stamp = self.stamp
        self.pub_img.publish(msg)

    def publish_cmd(self, steer, speed):
        adjusted_speed = self.speed_adj_multiple * speed

        # Max speed adjustment is needed for safety reasons
        if adjusted_speed > self.max_speed:
            adjusted_speed = self.max_speed

        # Min speed adjustment is needed so car would not stop on crossroads and instead move slowly forward
        if adjusted_speed < self.min_speed:
            adjusted_speed = self.min_speed

        msg = VehicleCmd()
        msg.header.stamp = self.stamp
        msg.gear_cmd = Gear(Gear.DRIVE)
        msg.ctrl_cmd.linear_velocity = adjusted_speed
        msg.ctrl_cmd.steering_angle = steer
        self.pub_cmd.publish(msg)

    def callback(self, front_wide_img):
        img = self.bridge.compressed_imgmsg_to_cv2(front_wide_img)
        if self.convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.crop(img)
        img = self.resize(img)
        self.stamp = front_wide_img.header.stamp
        self.publish_img(img)

        img = self.normalise(img)
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        self.img = np.array(img, dtype=np.float32, order='C')
        self.stamp = front_wide_img.header.stamp

    def pose_callback(self, current_pose):
        x = current_pose.pose.position.x
        y = current_pose.pose.position.y

        quaternion = [
            current_pose.pose.orientation.x, current_pose.pose.orientation.y,
            current_pose.pose.orientation.z, current_pose.pose.orientation.w
        ]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        speed, distance_error = self.speed_model.find_speed_for_position(x, y, yaw)
        if distance_error < self.max_closest_points_distance:
            self.speed = speed
        else:
            print("Correct vehicle speed not found: speed=" + str(speed) + " distance_error=" + str(distance_error))

    def turn_signal_callback(self, msg):
        self.turn_signal = int(msg.manual_input)

    def resize(self, img):
        return cv2.resize(img, dsize=(self.scaled_width, self.scaled_height), interpolation=cv2.INTER_LINEAR)

    def normalise(self, img):
        return (img / 255.0)

    def crop(self, img):
        return img[self.ymin:self.ymax, self.xmin:self.xmax, :]

    def run(self):
        print("NvidiaE2E ready!")
        while not rospy.is_shutdown():
            if self.img is not None:
                if self.model_type == "pilotnet-conditional":
                    steers = self.steering_model.predict(self.img, 3)
                    steer = steers[self.turn_signal]
                elif self.model_type == "pilotnet-control":
                    control = np.zeros(3, dtype=np.float32)
                    control[self.turn_signal] = 1.0
                    steer = self.steering_model.predict([self.img, control], 1)[0]
                elif self.model_type == "pilotnet":
                    steer = self.steering_model.predict(self.img)[0]
                elif self.model_type == "pilotnet-ebm":
                    steer = self.steering_model.predict(self.img)[0]
                else:
                    print("Unknown model type", self.model_type)
                    sys.exit()

                self.img = None
                self.publish_cmd(steer / self.steering_ratio, self.speed)

if __name__ == '__main__':
    rospy.init_node('nvidia_e2e', anonymous=True)
    node = NvidiaE2E()
    node.run()
