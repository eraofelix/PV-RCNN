import sys
import os
import numpy as np
import rospy
import logging
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from geometry_msgs.msg import Point
from autodrive_msgs.msg import Obstacles, Obstacle
from std_msgs.msg import Header
from kitti_utils import *
import time 


def make_simple_object(obj, calib):
    """For each obs, convert coordinates to velodyne frame.
    
    Args:
        obj ([type]): [description]
        calib ([type]): [description]
    
    Returns:
        [type]: [description]
    """
    xyz = calib.R0 @ obj.t              # (3,3)x(3,)=(3,), 
    xyz = calib.C2V @ np.r_[xyz, 1]     # (3,4)x(4,)=(3,),array([-0.86744172,  0.78475468, 25.00782068]), where np.r_[xyz, 1]: [x,y,z] ==> [x,y,z,1], 
    wlh = np.r_[obj.w, obj.l, obj.h]    # (3,), array([1.66, 3.2 , 1.61])， nothing changed???
    rz = np.r_[-obj.ry]                 # (1,), array([1.59])
    box = np.r_[xyz, wlh, rz]           # (7,), array([-0.86744172,  0.78475468, 25.00782068,  1.66,  3.2, 1.61,  1.59])
    obj = dict(box=box, class_idx=obj.class_idx)
    return obj

def make_simple_objects(item):
    objects = [make_simple_object(
        obj, item['calib']) for obj in item['objects']]
    item['boxes'] = np.stack([obj['box'] for obj in objects])
    item['class_idx'] = np.r_[[obj['class_idx'] for obj in objects]]

class KittiViewer():

    def __init__(self, root_dir):
        self.__root_dir = root_dir
        self.__bin_pub = rospy.Publisher('/bin', PointCloud2, queue_size=2)
        self.__obs_pub = rospy.Publisher('/gt', Obstacles, queue_size=10)
        self.__fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('rgba', 12, PointField.FLOAT32, 1)]
        self.__header = Header()
        self.__header.frame_id = "map"
        rospy.init_node('bin_publisher', anonymous=True)

    def __publish(self, idx):
        velo_path = os.path.join(self.__root_dir, 'velodyne_reduced', f'{idx:06d}.bin')
        calib = read_calib(os.path.join(self.__root_dir, 'calib', f'{idx:06d}.txt'))
        objects = read_label(os.path.join(self.__root_dir, 'label_2', f'{idx:06d}.txt'))
        item = dict(velo_path=velo_path, calib=calib, objects=objects, idx=idx)
        make_simple_objects(item)
        """item: dict
        {'velo_path': '/home/kun.fan/codes/PV-RCNN/data/kitti/training/velodyne_reduced/000007.bin', 
        'calib': <kitti_utils.Calibration object at 0x7f41b13f2080>, 
        'objects': [<kitti_utils.Object3d object at 0x7f41b13f2278>, 
                    <kitti_utils.Object3d object at 0x7f41b139bf98>, 
                    <kitti_utils.Object3d object at 0x7f41b139be48>, 
                    <kitti_utils.Object3d object at 0x7f41b139bd30>, 
                    <kitti_utils.Object3d object at 0x7f41b139beb8>, 
                    <kitti_utils.Object3d object at 0x7f41b139bcf8>], 
        'idx': 7, 
        'boxes': array([[ 2.52823562e+01,  1.05417258e+00, -4.86114994e-01,
                          1.66000000e+00,  3.20000000e+00,  1.61000000e+00,
                          1.59000000e+00],
                        [ 4.77217209e+01,  8.12777851e+00, -4.13867422e-01,
                          1.51000000e+00,  3.70000000e+00,  1.40000000e+00,
                          -1.55000000e+00],
                        [ 6.07242843e+01,  5.60410971e+00,  5.91432889e-02,
                          1.66000000e+00,  4.05000000e+00,  1.46000000e+00,
                          -1.56000000e+00],
                        [ 3.41857045e+01,  1.31270019e+01, -5.59530590e-01,
                          5.00000000e-01,  1.95000000e+00,  1.72000000e+00,
                          -1.54000000e+00],
                        [ -1.03343859e+03,  9.93951734e+02,  9.70750441e+02,
                          -1.00000000e+00, -1.00000000e+00, -1.00000000e+00,
                          1.00000000e+01],
                        [ -1.03343859e+03,  9.93951734e+02,  9.70750441e+02,
                          -1.00000000e+00, -1.00000000e+00, -1.00000000e+00,
                          1.00000000e+01]]), 
        'class_idx': array([ 0,  0,  0,  2, -1, -1])}
        """

        # for point cloud
        points = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
        cloud = point_cloud2.create_cloud(self.__header, self.__fields, points)

        obss = Obstacles()
        obss.header = self.__header
        for i in range(item['boxes'].shape[0]):
            boxes = item['boxes'][i]
            obs = Obstacle()
            obs.header = self.__header
            obs.ObsId = i
            obs.ObsPosition.x = boxes[0]
            obs.ObsPosition.y = boxes[1]
            obs.ObsPosition.z = boxes[2]
            obs.ObsTheta = boxes[6]
            obs.Length = boxes[4]
            obs.Width = boxes[3]
            obs.Height = boxes[5]

            pp1, pp2, pp3, pp4 = Point(), Point(), Point(), Point()
            pp1.z = pp2.z = pp3.z = pp4.z = obs.ObsPosition.z
            pp1.x = obs.ObsPosition.x - obs.Width/2.0 * np.sin(obs.ObsTheta) - obs.Length / 2.0 * np.cos(obs.ObsTheta)
            pp1.y = obs.ObsPosition.y + obs.Width/2.0 * np.cos(obs.ObsTheta) - obs.Length / 2.0 * np.sin(obs.ObsTheta)

            pp2.x = obs.ObsPosition.x + obs.Width/2.0 * np.sin(obs.ObsTheta) - obs.Length / 2.0 * np.cos(obs.ObsTheta)
            pp2.y = obs.ObsPosition.y - obs.Width/2.0 * np.cos(obs.ObsTheta) - obs.Length / 2.0 * np.sin(obs.ObsTheta)

            pp3.x = obs.ObsPosition.x + obs.Width/2.0 * np.sin(obs.ObsTheta) + obs.Length / 2.0 * np.cos(obs.ObsTheta)
            pp3.y = obs.ObsPosition.y - obs.Width/2.0 * np.cos(obs.ObsTheta) + obs.Length / 2.0 * np.sin(obs.ObsTheta)

            pp4.x = obs.ObsPosition.x - obs.Width/2.0 * np.sin(obs.ObsTheta) + obs.Length / 2.0 * np.cos(obs.ObsTheta)
            pp4.y = obs.ObsPosition.y + obs.Width/2.0 * np.cos(obs.ObsTheta) + obs.Length / 2.0 * np.sin(obs.ObsTheta)
            obs.PolygonPoints += [pp1, pp2, pp3, pp4]

            obss.obs.append(obs)

        count = 0    
        while count < 1:
            self.__bin_pub.publish(cloud)
            self.__obs_pub.publish(obss)
            time.sleep(0.5)
            count += 1

    def run(self):
        for idx in range(10000):
            print('idx={}'.format(idx))
            self.__publish(idx=idx)

if __name__ == '__main__':
    root_path = '/home/kun.fan/codes/PV-RCNN/data/kitti/training'
    kv = KittiViewer(root_path)
    kv.run()



