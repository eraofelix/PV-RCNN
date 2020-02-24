import sys
import os
import numpy as np
import rospy
import logging
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from autodrive_msgs.msg import Obstacles, Obstacle
from std_msgs.msg import Header
import time 


class KittiViewer():

    def __init__(self, root_dir):
        self.__root_dir = root_dir
        self.__bin_pub = rospy.Publisher('/bin', PointCloud2, queue_size=2)
        self.__obs_pub = rospy.Publisher('/gt_objects', Obstacles, queue_size=10)
        self.__fields = [PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('rgba', 12, PointField.FLOAT32, 1)]
        self.__header = Header()
        self.__header.frame_id = "map"
        rospy.init_node('bin_publisher', anonymous=True)

    def __publish(self, bin_path=None, label_path=None):
        # for point cloud
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        cloud = point_cloud2.create_cloud(self.__header, self.__fields, points)

        # for gt obs
        if label_path:
            obss = Obstacles()
            obss.header = self.__header
            # read label txt
            for i in range(num_obs):
                obs = Obstacle()
                obs.header = data.header
                obs.ObsId = i
                obs.ObsPosition.x = boxes_lidar[i,1]
                obs.ObsPosition.y = -boxes_lidar[i,0]
                obs.ObsPosition.z = -boxes_lidar[i,2]
                obs.ObsTheta = -boxes_lidar[i,6] + 3.1415926
                obs.Length = boxes_lidar[i,4]
                obs.Width = boxes_lidar[i,3]
                obs.Height = boxes_lidar[i,5]

                pp1 = Point()
                pp1.x = obs.ObsPosition.x - obs.Width/2.0 * np.sin(obs.ObsTheta) - obs.Length / 2.0 * np.cos(obs.ObsTheta)
                pp1.y = obs.ObsPosition.y + obs.Width/2.0 * np.cos(obs.ObsTheta) - obs.Length / 2.0 * np.sin(obs.ObsTheta)
                pp1.z = obs.ObsPosition.z
                obs.PolygonPoints.append(pp1)

                pp2 = Point()
                pp2.x = obs.ObsPosition.x + obs.Width/2.0 * np.sin(obs.ObsTheta) - obs.Length / 2.0 * np.cos(obs.ObsTheta)
                pp2.y = obs.ObsPosition.y - obs.Width/2.0 * np.cos(obs.ObsTheta) - obs.Length / 2.0 * np.sin(obs.ObsTheta)
                pp2.z = obs.ObsPosition.z
                obs.PolygonPoints.append(pp2)

                pp3 = Point()
                pp3.x = obs.ObsPosition.x + obs.Width/2.0 * np.sin(obs.ObsTheta) + obs.Length / 2.0 * np.cos(obs.ObsTheta)
                pp3.y = obs.ObsPosition.y - obs.Width/2.0 * np.cos(obs.ObsTheta) + obs.Length / 2.0 * np.sin(obs.ObsTheta)
                pp3.z = obs.ObsPosition.z
                obs.PolygonPoints.append(pp3)

                pp4 = Point()
                pp4.x = obs.ObsPosition.x - obs.Width/2.0 * np.sin(obs.ObsTheta) + obs.Length / 2.0 * np.cos(obs.ObsTheta)
                pp4.y = obs.ObsPosition.y + obs.Width/2.0 * np.cos(obs.ObsTheta) + obs.Length / 2.0 * np.sin(obs.ObsTheta)
                pp4.z = obs.ObsPosition.z
                obs.PolygonPoints.append(pp4)

                obss.obs.append(obs)

            
        while not rospy.core.is_shutdown():
            self.__bin_pub.publish(cloud)
            if label_path:
                self.__obs_pub.publish(obss)
            time.sleep(1)

    def run(self):
        bin_path = os.path.join(self.__root_dir, 'training/velodyne/006740.bin')
        self.__publish(bin_path=bin_path)

if __name__ == '__main__':
    root_path = '/mnt/data-2/data/kun.fan/public_datasets'
    kv = KittiViewer(root_path)
    kv.run()



