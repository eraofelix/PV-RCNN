import sys
import os
import numpy as np
import rospy
import logging
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
import time 


def use_mayavi(bin_path):
    print(sys.path)
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import mayavi.mlab
    import pykitti  # install using pip install pykitti
    velo = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
    mayavi.mlab.points3d(
        velo[:, 0],   # x
        velo[:, 1],   # y
        velo[:, 2],   # z
        velo[:, 2],   # Height data used for shading
        mode="point", # How to render each point {'point', 'sphere' , 'cube' }
        colormap='spectral',  # 'bone', 'copper',
        #color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
        scale_factor=100,     # scale of the points
        line_width=10,        # Scale of the line, if any
        figure=fig,
    )
    # velo[:, 3], # reflectance values
    mayavi.mlab.show()

def bin2ros(bin_path):
    velo = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)  # .astype(np.float16)
    rospy.init_node('bin_publisher', anonymous=True)
    pub = rospy.Publisher('/bin', PointCloud2, queue_size=2)
    
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              PointField('rgba', 12, PointField.FLOAT32, 1)]
    header = Header()
    header.frame_id = "map"
    cloud = point_cloud2.create_cloud(header, fields, velo)
    # import pdb;pdb.set_trace()
    while not rospy.core.is_shutdown():
        print('publishing.....')
        pub.publish(cloud)
        time.sleep(1)


if __name__ == '__main__':
    bin_path = '/mnt/data-2/data/kun.fan/public_datasets/training/velodyne/006740.bin'
    bin2ros(bin_path)


