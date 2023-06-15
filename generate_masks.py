from enum import unique
import string
from matplotlib import colors
from yaml import Loader 
from yaml import load as load_yaml
import numpy as np
from stl import mesh
import copy
import cv2
import transforms3d
from math import pi, ceil
from scipy.spatial.transform import Rotation
import logging
import os
import sys
from organize_dataset import organize_data
import argparse

def triangle_side_sign(q, a, b, c):
    '''
        Calculates the sign of the angle of (q-a)x(q-b) and (q-c).
        
        arguments:
            - q (tuple): coordinates of a point
            - a, b, c (tuples): coordinates of the points of a triangle
        output:
            - True/False: if the angle is positive/negative
    '''
    return (np.dot(np.cross((q-a),(q-b)),(q-c)) >= 0)

def triangle_line_intersect(q1, q2, a, b, c):
    '''
        Calculates the intersection of a triangle and a segment.
        
        arguments:
            - q1, q2 (tuples): coordinates of the two points of the segment
            - a, b, c (tuples): coordinates of the points of a triangle
        output:
            - True/False: if the segment intersects/does not intersect the triangle
    '''
    if triangle_side_sign(q1, a, b, c) != triangle_side_sign(q2, a, b, c):
        # if the signs are not equal
        side_signs = [triangle_side_sign(q1, q2, a, b), triangle_side_sign(q1, q2, b, c), triangle_side_sign(q1, q2, c, a)]
        if side_signs.count(triangle_side_sign(q1, q2, a, b)) == 3:
            return True
        else:
            return False
    else:
        # if the signs are equal, there is not intersection
        return False

def triangle_triangle_intersect(triangle_1, triangle_2):
    '''
        Calculates the intersection of two triangle.
        
        arguments:
            - triangle1, triangle2 (list of tuples): three coordinates of the triangles
        output:
            - True/False: if triangle1 intersects/does not intersect triangle2
    '''
    # get the corners of the triangles
    a1, b1, c1 = triangle_1
    a2, b2, c2 = triangle_2
    if triangle_line_intersect(a1, b1, a2, b2, c2):
        return True
    elif triangle_line_intersect(a1, c1, a2, b2, c2):
        return True
    elif triangle_line_intersect(b1, c1, a2, b2, c2):
        return True
    else:
        return False

def mesh_triangle_covered(mesh_triangle, camera_pose, other_triangle):
    '''
        Calculates that other_triangle covers mesh_triangle from camera.
        
        arguments:
            - triangle1, triangle2 (list of tuples): three coordinates of the triangles
        output:
            - True/False: if other_triangle covers/does not cover mesh_triangle from camera.
    '''
    covered = False
    for i in range(len(mesh_triangle)):
        side_triangle = copy.copy(mesh_triangle)
        side_triangle[i] = camera_pose
        # first: triangle, second: tetrahedron side
        if triangle_triangle_intersect(other_triangle, side_triangle):
            covered = True
            break
    return covered

def draw_meshes(mesh_paths, colors, poses, cam_mtx, dist, output_folder):
    '''
        Draws the masks of objects according to the overlaps.
        
        arguments:
            - mesh_paths (list[str]): file path of .stl models of the objects
            - colors (list[list[int]]): unique colors of the masks of the models (BGR)
            - poses (list[list]): positions and orientations of objects in the camera frame (in mm and Euler angles)
    '''
    meshes = []
    # select the unique object file names
    unique_path_names = list(np.unique(np.array(mesh_paths)))
    for mesh_path in unique_path_names:
        meshes.append(mesh.Mesh.from_file(mesh_path))
    
    mask = np.zeros((1080,1920,5))
    # This loop iterates all the objects
    for mesh_path, color, pose, object_id in zip(mesh_paths, colors, poses, range(len(mesh_path))):
        color = copy.copy(color)
        current_mesh = meshes[unique_path_names.index(mesh_path)]
        # assigns an object id to the items
        color.append(object_id)

        triangle_index = 0
        # This loop iterates all the triangles of the current object.
        for triangle, normal in zip(current_mesh.vectors, current_mesh.normals):            
            # Gets the orientation from pose
            euler_angles = pose[3:]
            # Calculates the transformation matrix and the rotation matrix between the camera and the object
            camera_to_object_transform = transforms3d.affines.compose(pose[0:3], transforms3d.euler.euler2mat(*euler_angles, axes='sxyz'), [1, 1, 1])
            camera_to_object_rotation = transforms3d.affines.compose([0,0,0], transforms3d.euler.euler2mat(*euler_angles, axes='sxyz'), [1, 1, 1])
            # Calculates the axis angle for robot from Euler angles
            rotvec_object = transforms3d.euler.euler2axangle(*pose[3:])[0] * transforms3d.euler.euler2axangle(*pose[3:])[1]
            
            # Calculate the ,coordinates of triangle center
            triangle_center = list(np.average(np.array(triangle), axis=0))
            triangle_center.append(1)
            normal = list(normal)
            normal.append(1)
            
            # Calculate the triangle center vector and triangle normal in camera
            triangle_center_in_camera = np.dot(np.array(camera_to_object_transform), triangle_center)
            triangle_normal_in_camera = np.dot(np.array(camera_to_object_rotation), normal)

            # This condition checks the angle between the normal of the current triangle and the vector from camera to triangle center.
            # If the angle is greather than pi/2 the triangle is non visible from camera so we discard it. We check it with dot product.
            if np.dot(triangle_center_in_camera, triangle_normal_in_camera) < 0.8:
                # The points list contains the projected coordinates of the current triangle from 3D to 2D
                points = project_3d_to_2d(triangle, (pose[0:3], rotvec_object), cam_mtx, dist)
                points = np.reshape(points,(3,2))
                points = np.array([(int(point[0]),int(point[1])) for point in points])
                # Calculate the bounding box of current triangle
                max_x = points[:,0].max()
                min_x = points[:,0].min()
                max_y = points[:,1].max()
                min_y = points[:,1].min()
                # Shift the points of current triangle with mimimum values of the coordinates
                points_scaled = np.array([(x-min_x,y-min_y) for (x,y) in points])
                # Create a list from the pixels in the bounding box
                indices = np.where(cv2.drawContours(np.zeros((max_y-min_y, max_x-min_x)), [points_scaled], 0, 255, -1) != 0)
                # Shift back the points to the original position
                indices_scaled = [(x+min_x,y+min_y) for (x,y) in zip(indices[1], indices[0])]
                # Create a dictionary for conflicting pixels
                conflicting_pixels = {}
                # This loop iterates on the pixels where the pixels already have color
                for x,y in indices_scaled:
                    # If the pixels are inside the image
                    if x > 0 and y > 0 and x < mask.shape[1] and y < mask.shape[0]:
                        # If none of the pixel in mask is colored in this position
                        if not (mask[y][x][0:3]).any():
                            color.append(triangle_index)
                            # Change the color
                            mask[y][x][:] = copy.copy(color)
                            color.pop(-1)
                        elif not list(mask[y][x][0:3]) == color[0:3]:
                            # create list from these values above holding the indices of triangles in the first object that intersect with our current triangle in the projected image
                            if (int(mask[y][x][3]),int(mask[y][x][4])) in conflicting_pixels:
                                conflicting_pixels[(int(mask[y][x][3]),int(mask[y][x][4]))].append((x,y))
                            else:
                                conflicting_pixels[(int(mask[y][x][3]),int(mask[y][x][4]))] = [(x,y)]

                # This loop checks triangle intersections where are conflicting pixels
                for key, value in conflicting_pixels.items():
                    o_id, t_id = key
                    # Select the triangle to which the conflicting pixel belongs
                    other_triangle = meshes[unique_path_names.index(mesh_paths[o_id])].vectors[t_id]
                    camera_to_other_object_transform = transforms3d.affines.compose(poses[o_id][0:3], transforms3d.euler.euler2mat(*poses[o_id][3:], axes='sxyz'), [1, 1, 1])
                    
                    # Transform the points of the other triangle to camera frame to compare the distance from camera with the current triangle
                    other_triangle_in_camera = []
                    for point in other_triangle:
                        point = list(point)
                        point.append(1)
                        other_triangle_in_camera.append(np.dot(np.array(camera_to_other_object_transform), point))
                    other_triangle_in_camera = np.array(other_triangle_in_camera)

                    # Transform the points of the current triangle to camera frame to compare the distance from camera with the other triangle
                    triangle_in_camera = []
                    for point in triangle:
                        point = list(point)
                        point.append(1)
                        triangle_in_camera.append(np.dot(np.array(camera_to_object_transform), point))
                    triangle_in_camera = np.array(triangle_in_camera)
                    
                    # Calculate the maximum and minimum distance of current triangle and another triangle from camera
                    triangle_max_distance = max(list(map(lambda p: np.linalg.norm(p), triangle_in_camera)))
                    other_triangle_min_distance = min(list(map(lambda p: np.linalg.norm(p), other_triangle_in_camera)))
                    triangle_min_distance = min(list(map(lambda p: np.linalg.norm(p), triangle_in_camera)))
                    other_triangle_max_distance = max(list(map(lambda p: np.linalg.norm(p), other_triangle_in_camera)))

                    if other_triangle_max_distance < triangle_min_distance:
                        # If the other triangle is above the current triangle, we do nothing.
                        pass
                    elif other_triangle_min_distance > triangle_max_distance:
                        # If the other triangle is under the current triangle, fill pixels of current triangle.
                        color.append(triangle_index)
                        for x,y in value:
                            mask[y][x][:] = copy.copy(color)
                        color.pop(-1)
                    else:
                        # Else check tetrahedron intersection, and fill pixels of current triangle.
                        if not mesh_triangle_covered(triangle_in_camera[:,0:3], [0,0,0], other_triangle_in_camera[:,0:3]):
                            color.append(triangle_index)
                            for x,y in value:
                                mask[y][x][:] = copy.copy(color)
                            color.pop(-1)

            # print(triangle_index)
            triangle_index += 1

    cv2.imwrite(os.path.join(output_folder, str(img_cnt).zfill(5) + '_annotation.png'), mask[:,:,0:3])
    return mask

def project_3d_to_2d(points_3d, frame_transform, cam_mtx, dist):
        '''
        Project 3D points onto the 2D image

        arguments:
         - points_3d (list): list of x,y,z, values, in the form of [[x,y,z]...]
         - frame_transform (tuple): tuple of form ((x,y,z), (rx,ry,rz)), the translation and rotation of the frame (relative to the camera frame) in which the "3d_points" are given

        returns:
         - points_2d (list): list of 2D points or None if camera is not calibrated
        '''
        if not cam_mtx is None and not dist is None:

            points_3d = np.array(points_3d).astype(np.float32)

            rx, ry, rz = frame_transform[1]
            rvec = np.array([[rx], [ry], [rz]]).astype(np.float32)

            x, y, z = frame_transform[0]
            tvec = np.array([[x], [y], [z]]).astype(np.float32)

            points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, cam_mtx, dist)
            return (points_2d)
        else:
            logging.warn('The camera matrix and the distrotion values are empty, either the reading of the config files failed or the camera is not calibrated!')
            return(None)

def get_object_pose_in_camera(object_pose_in_base, tcp_pose_in_base, camera_pose_in_tcp):
    '''
        Get object pose in camera frame.

        arguments:
            - object_pose_in_base: [x,y,z,rx,ry,rz] xyz in meters, rx, ry, rz in Euler angles (radians)
            - tcp_pose_in_base: [x,y,z,rx,ry,rz] xyz in meters, rx, ry, rz in rotation vector
            - camera_pose_in_tcp: transforms3d.affines.compose(...) transformation matrix, translation in meters

        return:
            - object_in_camera_pose: [x,y,z,rx,ry,rz] xyz in mm, rx, ry, rz in Euler angles (radians)
    '''
    object_pose = object_pose_in_base
    tcp_pose = tcp_pose_in_base

    object_xyz = list(map(lambda x: x*1000, object_pose[0:3]))
    object_xyz.extend(object_pose[3:])

    tcp_xyz = list(map(lambda x: x*1000, tcp_pose[0:3]))
    tcp_xyz.extend(Rotation.from_rotvec(np.array(tcp_pose[3:])).as_euler("xyz"))

    object_pose = object_xyz
    tcp_pose = tcp_xyz
    camera_in_tcp = camera_pose_in_tcp

    object_to_base_transform = transforms3d.affines.compose(object_pose[0:3], transforms3d.euler.euler2mat(*object_pose[3:], axes='sxyz'), [1, 1, 1])
    tcp_to_base_transform = transforms3d.affines.compose(tcp_pose[0:3], transforms3d.euler.euler2mat(*tcp_pose[3:], axes='sxyz'), [1, 1, 1])

    object_in_camera = np.dot(np.linalg.inv(camera_in_tcp), np.dot(np.linalg.inv(np.array(tcp_to_base_transform)), np.array(object_to_base_transform)))
    object_in_camera_translation = list(transforms3d.affines.decompose44(object_in_camera)[0])
    object_in_camera_rotation = transforms3d.euler.mat2euler(transforms3d.affines.decompose44(object_in_camera)[1])
    
    object_in_camera_pose = object_in_camera_translation
    object_in_camera_pose.extend(object_in_camera_rotation)

    return object_in_camera_pose

def vector_to_homogeneous(vector):
    '''
    vector: [x,y,z,rx,ry,rz]
    '''
    rotation_euler = Rotation.from_euler('xyz',vector[3:])
    translation = vector[0:3]
    rotation_matrix = rotation_euler.as_matrix()
    homogeneous_matrix = np.zeros((4,4))
    homogeneous_matrix[0:3,0:3] = rotation_matrix
    homogeneous_matrix[0:3,3] = translation
    homogeneous_matrix[3,3] = 1
    homogeneous_matrix = np.round(homogeneous_matrix,15)
    return homogeneous_matrix

def get_unique_colors(count, limit=0):
    '''
        Get unique HSV colors for masks. The hue value is the same for each class

        arguments:
            - count (int): number of classes
            - limit (int): minimum saturation value
        returns:
            - colors: list of HSV colors
    '''
    colors = []
    increment = int((255-limit)/count)

    for v in range(count):
        colors.append(v*increment+limit)

    return colors

# Define argument parser
parser = argparse.ArgumentParser(description='Generate segmentation masks from 3D models and camera object poses')
parser.add_argument('-p', '--project', type=str, dest='project', help='Name of project', required=True)
parser.add_argument('--organize', dest='organize', action='store_true', default=False, help='Organize dataset')
args = parser.parse_args()

if __name__=='__main__':
    # Make directory
    PROJECT_FOLDER  = 'projects'
    OUTPUT_FOLDER = 'annotations'
    project = args.project
    os.makedirs(os.path.join(PROJECT_FOLDER, project, OUTPUT_FOLDER), exist_ok=True)

    # Open configuration file
    try:
        with open(os.path.join(PROJECT_FOLDER, project, 'config.yaml'), 'r') as f:
            configs = load_yaml(f, Loader=Loader)
    except FileNotFoundError as e:
        logging.error(e)
        sys.exit(1)

    # Get variables from configuration file
    robot_tcp_poses_csv = configs['robot_tcp_poses_csv']
    cam_mtx = np.array(configs['camera_matrix'])
    dist = np.array(configs['dist'])
    objects = configs['objects']
    camera_in_tcp_translation = configs['camera_in_tcp_translation']
    camera_in_tcp_rotation = configs['camera_in_tcp_rotation']

    # Load the .csv file
    try:
        tcp_photo_poses = np.genfromtxt(os.path.join(PROJECT_FOLDER, project, robot_tcp_poses_csv), delimiter=',')
    except FileNotFoundError as e:
        logging.error(e)
        sys.exit(1)

    # camera_in_tcp_translation = np.zeros(3)
    # camera_in_tcp_rotation = np.identity(3)
    camera_in_tcp = transforms3d.affines.compose(camera_in_tcp_translation, camera_in_tcp_rotation, [1,1,1])

    mesh_paths = []
    object_poses_in_camera = []
    possible_colors = []

    for img_cnt in range(len(tcp_photo_poses)):
        print(img_cnt+1, "/", len(tcp_photo_poses))
        tcp_photo_pose = tcp_photo_poses[img_cnt]

        # Convert quaternion to rotation vector
        axis, angle = transforms3d.quaternions.quat2axangle(np.array([tcp_photo_pose[6],tcp_photo_pose[3],tcp_photo_pose[4],tcp_photo_pose[5]])) # w,x,y,z!!!
        rotvec = axis*angle
        trans = tcp_photo_pose[0:3]

        unique_objects = []
        object_instances = []
        for object in objects:
            if object["stl_file"] not in unique_objects:
                unique_objects.append(object["stl_file"])
                object_instances.append(1)
            else:
                object_instances[unique_objects.index(object["stl_file"])] += 1
            object_poses_in_camera.append(get_object_pose_in_camera(object["pose"], np.append(trans,rotvec), camera_in_tcp))
            mesh_paths.append(os.path.join(PROJECT_FOLDER, project, object["stl_file"]))

        saturations = []
        for number_of_instances in object_instances:
            saturations.append(get_unique_colors(number_of_instances,limit=100))

        hues = get_unique_colors(len(unique_objects))
        instance_indices = np.zeros(np.array(unique_objects).shape).astype(np.uint32)
        for mesh_path in mesh_paths:
            index = unique_objects.index(os.path.split(mesh_path)[1])
            hue = hues[index]
            saturation_index = instance_indices[index]
            saturation = saturations[index][saturation_index]
            rgb = list(cv2.cvtColor(np.array([[[hue,saturation,255]]]).astype(np.uint8), cv2.COLOR_HSV2RGB)[0][0])
            possible_colors.append(rgb)
            instance_indices[index] += 1

        # Generate masks
        mask_final = draw_meshes(mesh_paths, possible_colors, object_poses_in_camera, cam_mtx, dist, os.path.join(PROJECT_FOLDER, project, OUTPUT_FOLDER))

        object_poses_in_camera = []
        mesh_paths = []

    # Organize the generated masks, training and validation data    
    if args.organize:
        try:
            organize_data(os.path.join(PROJECT_FOLDER, project, OUTPUT_FOLDER))
        except FileNotFoundError as e:
            logging.error(e)
            sys.exit(1)