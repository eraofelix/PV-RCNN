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
        self.__header.frame_id = "chassis_base"
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
        pred_boxes = np.array([[ 10.8142,  4.3419, -1.5002,  1.7520,  4.2997,  1.5199,  1.6309],
                        [17.6510, -2.2238, -1.6200,  1.7596,  4.2559,  1.5404,  1.6067],
                        [19.1842,  4.1860, -1.5994,  1.7713,  4.2742,  1.5262,  1.5747],
                        [20.6845, 26.0792, -1.8360,  1.7538,  4.3198,  1.5311,  1.6847],
                        [ 7.6023,  0.2244, -1.4496,  1.7894,  4.3437,  1.5543,  1.6053],
                        [ 2.9149,  4.2401, -1.4803,  1.7520,  4.7347,  1.6105,  1.6369],
                        [27.1094,  8.8551, -1.7229,  1.7546,  4.2266,  1.5490,  1.5938],
                        [49.2019,  5.6506, -2.0619,  1.7468,  4.3082,  1.5326,  1.6352],
                        [ 2.7610,  8.0643, -1.2577,  1.6359,  4.1076,  1.5095,  1.6099],
                        [38.1095, -1.5373, -1.8322,  1.8133,  4.3309,  1.5765,  1.6383],
                        [11.9913,  7.9866, -1.5131,  1.7195,  4.1635,  1.5327,  1.5961]])
        for i in range(pred_boxes.shape[0]):
            boxes = pred_boxes[0]
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
            print(pp1.x, pp1.y, pp2.x, pp2.y, pp3.x, pp3.y, pp4.x, pp4.y)

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



