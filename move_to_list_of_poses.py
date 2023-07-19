#! /usr/bin/env python

import rospy
import moveit_commander
import rospkg
import actionlib
from bark_msgs.msg import CameraProjectionsAction, CameraProjectionsGoal, RaspberryPiCameraAction, RaspberryPiCameraGoal
from cv_bridge import CvBridge
import geometry_msgs.msg
from sensor_msgs.msg import JointState
import copy
import tf
import tf2_ros
import sys
import copy
import os
import cv2
import numpy as np

class BarkSlidebot:
    def __init__(self, group_name="manipulator"):
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.group_name = group_name
        self.group = moveit_commander.MoveGroupCommander(self.group_name)
        rospy.sleep(0.5)
        self.configs = rospy.get_param("/bark_slidebot_config")
        self.camera_matrix = np.array(rospy.get_param("camera_matrix"))
        self.distortion_vector = np.array(rospy.get_param("distortion_vector"))

        self.camera_client = actionlib.SimpleActionClient('raspberry_pi_camera', RaspberryPiCameraAction)
        self.camera_client.wait_for_server()

        self.tf_listener = tf.TransformListener()
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        self.bridge = CvBridge()
        rospy.sleep(0.2)

    ##  Moves the specified frame to the specified position.
    #   @param pose The frame is specified in header.frame_id of PoseStamped and the position and orientation in pose of PoseStamped.
    def move_to_pose_with_frame(self, pose):
        '''
        Moves the specified frame to the specified position.

        Arguments:
            - pose (geometry_msgs/PoseStamped): The frame is specified in header.frame_id of PoseStamped and the position and orientation in pose of PoseStamped.
        '''
        ## We can get the name of the reference frame for this robot:
        transformer = tf.Transformer(True, rospy.Duration(10.0))
        transform = geometry_msgs.msg.TransformStamped()
        transform.header.frame_id = "world"
        transform.child_frame_id = "base_link"
        transform.transform.rotation.w = 1
        transformer.setTransform(transform)

        transform = geometry_msgs.msg.TransformStamped()
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "base"
        transform.transform.rotation.z = 1
        transformer.setTransform(transform)

        transform = geometry_msgs.msg.TransformStamped()
        transform.header.frame_id = "base"
        transform.child_frame_id = "target"
        transform.transform.translation.x = pose.pose.position.x
        transform.transform.translation.y = pose.pose.position.y
        transform.transform.translation.z = pose.pose.position.z
        transform.transform.rotation = pose.pose.orientation
        transformer.setTransform(transform)

        transform = geometry_msgs.msg.TransformStamped()
        transform.header.frame_id = "target"
        eef_link = self.group.get_end_effector_link()
        transform.child_frame_id = eef_link
        trans, rot = self.lookup_transform(eef_link, pose.header.frame_id)
        transform.transform.translation = geometry_msgs.msg.Vector3(*trans)
        transform.transform.rotation = geometry_msgs.msg.Quaternion(*rot)
        transformer.setTransform(transform)

        trans, rot = transformer.lookupTransform("base_link", eef_link, rospy.Time(0))
        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.position = geometry_msgs.msg.Point(*trans)
        pose_goal.orientation = geometry_msgs.msg.Quaternion(*rot)

        waypoints = []
        # waypoints.append(self.group.get_current_pose().pose)  # Don't put the start point to waypoints!
        waypoints.append(copy.deepcopy(pose_goal))

        (plan_trajectory, fraction) = self.group.compute_cartesian_path(
                             waypoints,   # waypoints to follow
                             0.01,        # eef_step
                             0.0)         # jump_threshold

        if fraction == 1:
            self.group.execute(plan_trajectory, wait = True)
            return True
        else:
            return False

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
    robot = BarkSlidebot()
    rospack = rospkg.RosPack()
    rospackage_root = rospack.get_path("bark_slidebot")

    input("Press ENTER to move start pose!")

    robot.move_to_joint_pose("start_jpose")

    with open(os.path.join(rospackage_root, 'data', 'generated_poses.csv'), mode='r') as f:

        # Iterate through the poses
        for i, line in enumerate(f):
            print(i, '/', len(f))
            pose = [float(val) for val in line.split(',')]
            pose_msg = geometry_msgs.msg.PoseStamped()
            pose_msg.header.frame_id = 'camera'
            pose_msg.pose.position = geometry_msgs.msg.Point(*pose[0:3])
            pose_msg.pose.orientation = geometry_msgs.msg.Quaternion(*pose[3:])

            if robot.move_to_pose_with_frame(pose_msg):
                robot.move_to_pose_with_frame(pose_msg)
                joint_pose = robot.group.get_current_joint_values()
                trans, rot = robot.lookup_transform("tool0", "base")
                cartesian_pose = [trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]]

                # Get image
                img = cv2.cvtColor(robot.get_image(), cv2.COLOR_RGB2BGR)

                # Save image
                cv2.imwrite(os.path.join(rospackage_root, 'data', 'images_in_poses', str(i).zfill(5) + '.png'), img)
