import numpy as np

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


class CalibPandar(object):
    def __init__(self, V2C, R0):
        self.V2C = V2C
        self.C2V = self.inverse_rigid_trans(self.V2C)
        self.R0 = R0

    def inverse_rigid_trans(self, Tr):
        """ Inverse a rigid body transform matrix (3x4 as [R|t])
            [R"|-R"t; 0|1]
        """
        inv_Tr = np.zeros_like(Tr) # 3x4
        inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
        inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
        return inv_Tr

def read_calib_pandar(calib_path):
    with open(calib_path) as f:
        lines = f.readlines()
    P1 = np.array(lines[0].strip().split(" ")[1:], dtype=np.float32).reshape(3, 4) # P1
    R1_rect = np.array(lines[1].strip().split(" ")[1:], dtype=np.float32).reshape(3, 3)
    R = np.array(lines[2].strip().split(" ")[1:], dtype=np.float32).reshape(3, 3)
    t = np.array(lines[3].strip().split(" ")[1:], dtype=np.float32).reshape(3,1)
    Rt = np.hstack([R,t])
    calib = CalibPandar(Rt, R1_rect)
    return calib
