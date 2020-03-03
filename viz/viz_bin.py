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
sys.path.append(os.path.join(os.sep.join(os.path.abspath(__file__).split("/")[:-2]),''))
from pvrcnn.dataset.kitti_utils import read_calib, read_label
from pvrcnn.dataset.pandar_utils import read_calib_pandar, make_simple_objects
import time 
class KittiViewer():
    def __init__(self, root_dir):
        self.__root_dir = root_dir
        self.__bin_pub = rospy.Publisher('/bin', PointCloud2, queue_size=2)
        self.__obs_pub = rospy.Publisher('/gt', Obstacles, queue_size=10)
        self.__obs_pred_pub = rospy.Publisher('/pred', Obstacles, queue_size=10)
        self.__fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('rgba', 12, PointField.FLOAT32, 1)]
        self.__header = Header()
        self.__header.frame_id = "map"
        rospy.init_node('bin_publisher', anonymous=True)

    def __publish(self, bin_name):
        velo_path = os.path.join(self.__root_dir, 'velodyne_reduced', '{}.bin'.format(bin_name))
        calib_path = os.path.join(self.__root_dir, 'calib', '{}.txt'.format(bin_name))
        label_path = os.path.join(self.__root_dir, 'label_2', '{}.txt'.format(bin_name))
        calib_lines = sum(1 for line in open(calib_path))
        if calib_lines < 7:
            calib = read_calib_pandar(calib_path)
        else:
            calib = read_calib(calib_path)
        objects = read_label(label_path)
        item = dict(velo_path=velo_path, calib=calib, objects=objects, idx=0)
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
            obs.ObsTheta = boxes[6] + np.pi / 2
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

        obss_pred = Obstacles()
        obss_pred.header = self.__header
        pred_boxes = np.array([[ 7.6149, -3.6703, -1.9668,  1.7707,  4.6234,  1.7978,  3.1490],
        [10.4535, -4.3391, -1.6602,  1.6664,  4.5384,  1.6404,  2.5522],
        [ 9.2430, -3.9556, -1.7602,  1.7009,  4.5118,  1.6653,  2.0695],
        [ 8.4391, -1.8341, -1.8057,  1.7468,  4.5431,  1.7306,  3.4417],
        [ 7.6652,  4.7613, -1.3137,  1.7320,  4.6263,  1.9815,  1.8110]])
        for i in range(pred_boxes.shape[0]):
            boxes = pred_boxes[i]
            obs = Obstacle()
            obs.header = self.__header
            obs.ObsId = i
            obs.ObsPosition.x = boxes[0]
            obs.ObsPosition.y = boxes[1]
            obs.ObsPosition.z = boxes[2]
            obs.ObsTheta = boxes[6] + np.pi / 2
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

            obss_pred.obs.append(obs)

        count = 0    
        while count < 1000:
            self.__bin_pub.publish(cloud)
            self.__obs_pub.publish(obss)
            self.__obs_pred_pub.publish(obss_pred)
            time.sleep(1)
            count += 1

    def run(self):
        # bin_name = f'{idx:06d}.txt'
        bin_name = '1544426448586'
        print('bin_name={}'.format(bin_name))
        self.__publish(bin_name=bin_name)

if __name__ == '__main__':
    root_path = '/home/kun.fan/codes/PV-RCNN/data/kitti/training'
    kv = KittiViewer(root_path)
    kv.run()



