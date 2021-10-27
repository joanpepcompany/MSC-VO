import sys
import os
import numpy as np
import argparse
import yaml
import cv2

import rospy
from tf import TransformBroadcaster
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
from rospy import Time 

#change these to fit the expected topic names
IMAGE_MESSAGE_TOPIC = 'image_raw'
CAMERA_MESSAGE_TOPIC = 'camera_info'

# Depth image scale. ICL_NUIM dataset 5.0 others 1.0
DEPTH_SCALE = 1.0
# Frequency to publish the data. 
PUBLISH_RATE = 3

def yaml_to_CameraInfo(yaml_fname):
    """
    Parse a yaml file containing camera calibration data (as produced by 
    rosrun camera_calibration cameracalibrator.py) into a 
    sensor_msgs/CameraInfo msg.
    
    Parameters
    ----------
    yaml_fname : str
        Path to yaml file containing camera calibration data
    Returns
    -------
    camera_info_msg : sensor_msgs.msg.CameraInfo
        A sensor_msgs.msg.CameraInfo message containing the camera calibration
        data
    """
    # Load data from file
    
    with open(yaml_fname, 'r') as stream:
        calib_data = yaml.safe_load(stream)
    
    # Parse
    fx = calib_data["Camera.fx"]
    fy = calib_data["Camera.fy"]
    cx = calib_data["Camera.cx"]
    cy = calib_data["Camera.cy"]

    k1 = calib_data["Camera.k1"]
    k2 = calib_data["Camera.k2"]
    p1 = calib_data["Camera.p1"]
    p2 = calib_data["Camera.p2"]
    k3 = 0

    # Camera matrices
    K = [fx, 0.0, cx, 0.0, fy, cy, 0.0 , 0.0, 1.0]
    D = [float(k1), float(k2),float(p1), float(p2),float(k3)]
    R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]

    camera_info_msg = CameraInfo()
    camera_info_msg.width = calib_data["Camera.width"]
    camera_info_msg.height = calib_data["Camera.height"]
    camera_info_msg.K = K
    camera_info_msg.D = D
    camera_info_msg.R = R
    camera_info_msg.P = P
    camera_info_msg.distortion_model = "plumb_bob"
    return camera_info_msg


def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 

    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]

    return dict(list)


def associate(first_list, second_list,offset,max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))    
    """
    first_keys = first_list.keys()
    second_keys = second_list.keys()
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()

    associate_list = []
    for a,b in matches:
        img_list[a].insert(0, a)
        associate_list.append (img_list[a] + (pose_list[b]))
        
    return associate_list

   
def publish_data(associate_list, img_file, camera_info_msg):
    rospy.init_node('img_n_pose_publisher')

    # Get path to images (remove associate.txt from teh entire path)
    last_char_index = img_file.rfind("/")
    relative_path = img_file[:last_char_index] 
    
    tf_publisher = TransformBroadcaster()

    rgb_img_pub = rospy.Publisher('camera/color' + '/' + IMAGE_MESSAGE_TOPIC, Image, queue_size=10)
    rgb_cam_pub = rospy.Publisher('camera/color' + '/' + CAMERA_MESSAGE_TOPIC, CameraInfo, queue_size=10)
    depth_img_pub = rospy.Publisher('camera/depth' + '/' + IMAGE_MESSAGE_TOPIC, Image, queue_size=10)
    depth_cam_pub = rospy.Publisher('camera/depth' + '/' + CAMERA_MESSAGE_TOPIC, CameraInfo, queue_size=10)


    while not rospy.is_shutdown():
        for j in associate_list:
            ## Publish Pose
            translation = (float(j[4]), float(j[5]), float(j[6]) )
            rotation = (float(j[7]), float(j[8]), float(j[9]), float(j[10]) )
            timestamp = float(j[0])
    
            print('Timestamp from file: ', timestamp)
            rate = rospy.Rate(PUBLISH_RATE)  # hz
            timestamp =  Time.now()
        
            tf_publisher.sendTransform(translation, rotation,  timestamp, 'camera_color_optical_frame', 'odom')

            ### Publish Images    
            # Get images path       
            rgb_path =  relative_path + '/' +j[1]
            depth_path =  relative_path + '/' +j[3]

            # Load images            
            rgb_img = cv2.imread(rgb_path)
            depth_img = cv2.imread( depth_path, -1 )  
            # Depth Scale Factor ICL-NUIM 5, TUM and real-sense 1
            depth_img = depth_img/DEPTH_SCALE


            # Configure msgs
            bridge = CvBridge()
            rgb_img_msg = bridge.cv2_to_imgmsg(rgb_img, encoding="passthrough")
            rgb_img_msg.height = rgb_img.shape[0]
            rgb_img_msg.width = rgb_img.shape[1]
            rgb_img_msg.step = rgb_img.strides[0] 
            rgb_img_msg.encoding = 'bgr8'
            rgb_img_msg.header.frame_id = 'camera_color_optical_frame'
            rgb_img_msg.header.stamp = timestamp

            depth_img_msg =bridge.cv2_to_imgmsg(depth_img,encoding="passthrough")
            depth_img_msg.height = depth_img.shape[0]
            depth_img_msg.width = depth_img.shape[1]
            depth_img_msg.step = depth_img.strides[0]
            depth_img_msg.header.frame_id = 'camera_color_optical_frame'
            depth_img_msg.header.stamp = timestamp

            # Update camera info timest
            camera_info_msg.header.stamp = timestamp
            camera_info_msg.header.frame_id = "camera_color_optical_frame"

            rgb_img_pub.publish(rgb_img_msg)
            depth_img_pub.publish(depth_img_msg)
            rgb_cam_pub.publish(camera_info_msg)
            depth_cam_pub.publish(camera_info_msg)
            rate.sleep()
    

if __name__ == '__main__':
    
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script takes two data files with timestamps and associates them   
    ''')
    parser.add_argument('img_file', help='Images text file (Associate.txt in TUM format: timestamp rgb_img timestamp depth_img)')
    parser.add_argument('poses_file', help='Poses text file (TUM format: timestamp x, y, z, q0, q1, q2, q3)')
    parser.add_argument('calib_file', help='Camera calibration yaml file')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 0.02)',default=0.02)
    args = parser.parse_args()
    
    # Read files
    img_list = read_file_list(args.img_file)
    pose_list = read_file_list(args.poses_file)

    # Parse yaml file
    camera_info_msg = yaml_to_CameraInfo(args.calib_file)

    # Associate images with poses
    associate_list = associate(img_list, pose_list,float(args.offset),float(args.max_difference))  

    # Publish images, calibration data and poses
    publish_data(associate_list, args.img_file, camera_info_msg)  
   