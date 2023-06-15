#! /usr/bin/env python

import rospy
import moveit_commander
import rospkg
import actionlib
from bark_msgs.msg import RaspberryPiCameraAction, RaspberryPiCameraGoal
from cv_bridge import CvBridge
from sensor_msgs.msg import JointState
import tf
import tf2_ros
import sys
import os
from math import pi
import cv2
import numpy as np
import csv

class FreedriveRobot:
    def __init__(self, group_name="manipulator"):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.group_name = group_name
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        rospy.sleep(0.5)
        self.configs = rospy.get_param("/bark_slidebot_config")
        # self.camera_matrix = np.array(rospy.get_param("camera_matrix"))
        # self.distortion_vector = np.array(rospy.get_param("distortion_vector"))

        self.camera_client = actionlib.SimpleActionClient('raspberry_pi_camera', RaspberryPiCameraAction)
        self.camera_client.wait_for_server()

        self.tf_listener = tf.TransformListener()
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.bridge = CvBridge()

        # table_pose_stamped = geometry_msgs.msg.PoseStamped()
        # table_pose_stamped.header.frame_id = self.robot.get_planning_frame()
        rospy.sleep(0.2)

    ##  Gives the translation and orientation of child frame in parent frame.
    #   @param parent Name of the parent frame.
    #   @param child Name of the child frame.
    #   @return trans The translation (x, y, z in meter) of child frame in parent frame.
    #   @return rot The orientation of child frame in parent frame represented in quaternion (x, y, z, w).
    #   @return Retruns 'None' if the parent or child frame is not exist.
    def lookup_transform(self, parent, child):
        '''
        Gives the translation and orientation of child frame in parent frame.

        Arguments:
            - parent, child (string): name of the parent and child frames
        
        Returns:
            - trans (list): the translation (x, y, z in meter) of child frame in parent frame
            - rot (quaternion): the orientation of child frame in parent frame represented in quaternion (x, y, z, w)
            - Retruns 'None' if the parent or child frame is not exist.
        '''
        if self.tf_listener.frameExists(parent) and self.tf_listener.frameExists(child):
            t = self.tf_listener.getLatestCommonTime(child, parent)
            (trans, rot) = self.tf_listener.lookupTransform(child, parent, t)
            return (trans, rot)
        else:
            return None

    ##  Moves the robot to a specified joint pose.
    #   @param pose_name Name of the pose where move the robot. The joint variables come from the config file by name.
    def move_to_joint_pose(self, pose_name):
        '''
        Moves the robot to a specified joint pose.

        Arguments:
         - pose_name (string): name of the pose where move the robot. The joint variables come from the config file by name.
        '''
        photo_pose = self.configs[pose_name]

        joint_goal = JointState()
        joint_goal.name = self.group.get_active_joints()
        joint_values = photo_pose
        joint_goal.position = joint_values

        if not joint_values is None:
            success, plan_trajectory, planning_time, error_code = self.group.plan(joint_goal)
            if success:
                self.group.execute(plan_trajectory, wait = True)

    ##  Gets image from Raspberry Pi camera.
    #   @param self The object pointer.
    #   @return The captured image.
    def get_image(self):
        '''
        Gets image from Raspberry Pi camera.

        Returns:
            - (cv::Mat): OpenCV image. The captured image.
        '''
        ## Instantiate the goal
        goal = RaspberryPiCameraGoal()
        ## Send the goal and wait for the result
        self.camera_client.send_goal(goal)
        self.camera_client.wait_for_result(rospy.Duration.from_sec(30.0))

        return self.bridge.imgmsg_to_cv2(self.camera_client.get_result().image, "bgr8")

if __name__=='__main__':
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('bark_slidebot', anonymous=True)
    robot = FreedriveRobot()
    rospack = rospkg.RosPack()
    rospackage_root = rospack.get_path("bark_slidebot")

    input("Press ENTER to move start pose!")
    robot.move_to_joint_pose("start_jpose_up")

    with open(os.path.join(rospackage_root, 'data', 'freedrive_images', 'freedrive_poses.csv'), mode='a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
    
        i = 0
        while True:  
            input("Press ENTER to get cartesian pose, and save image!")

            # Get image
            img = cv2.cvtColor(robot.get_image(), cv2.COLOR_RGB2BGR)
            # Save image
            cv2.imwrite(os.path.join(rospackage_root, 'data', 'freedrive_images', str(i).zfill(5) + '.png'), img)
            # Get cartesian pose
            trans, rot = robot.lookup_transform("tool0", "base")
            cartesian_pose = [trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]]

            writer.writerow(cartesian_pose)
            
            print(str(i+1) + ". image and pose saved")
            i += 1