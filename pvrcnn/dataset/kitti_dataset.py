from tqdm import tqdm
import pickle
import numpy as np
import torch
import os
from copy import deepcopy
import os.path as osp
import os
from torch.utils.data import Dataset

from pvrcnn.core import ProposalTargetAssigner, AnchorGenerator
from .kitti_utils import read_calib, read_label, read_velo
from .pandar_utils import read_calib_pandar
from .augmentation import ChainedAugmentation
from .database_sampler import DatabaseBuilder


class KittiDataset(Dataset):

    def __init__(self, cfg, split='val'):
        super(KittiDataset, self).__init__()
        self.cfg = cfg
        self.split = split
        self.load_annotations(cfg)

    def __len__(self):
        return len(self.inds)

    def read_splitfile(self, cfg):
        if cfg.DATA.DATASET == 'pandar':
            bins_dir = osp.join(cfg.DATA.ROOTDIR, 'velodyne_reduced')
            self.inds = [n[:-4] for n in os.listdir(bins_dir) if '.bin' in n]
        else:  # kitti
            fpath = osp.join(cfg.DATA.SPLITDIR, f'{self.split}.txt')
            self.inds = np.loadtxt(fpath, dtype=np.int32).tolist()

    def read_cached_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, f'{self.split}.pkl')
        with open(fpath, 'rb') as f:
            self.annotations = pickle.load(f)
        print(f'Found cached annotations: {fpath}')

    def cache_annotations(self, cfg):
        fpath = osp.join(cfg.DATA.CACHEDIR, f'{self.split}.pkl')
        with open(fpath, 'wb') as f:
            pickle.dump(self.annotations, f)

    def create_annotation(self, idx, cfg):
        """create_annotation

        data example:
        
        idx = 7
        velo_path = '../data/kitti/training/velodyne_reduced/000007.bin'
        calib = <pvrcnn.dataset.kitti_utils.Calibration object at 0x7fcc7da2e750>
            calib.P   : 3x4, line P2 in txt
            calib.C2V : 3x4, line Tr_velo_to_cam in txt
            calib.V2C : 3x4, inverse_rigid_trans(C2V)
            calib.R0  : 3x3, line R0_rect in txt
        objects = [ <pvrcnn.dataset.kitti_utils.Object3d object at 0x7fcc7da87150>, 
                    <pvrcnn.dataset.kitti_utils.Object3d object at 0x7fcc7d9f8950>, 
                    <pvrcnn.dataset.kitti_utils.Object3d object at 0x7fcc7d9f89d0>, 
                    <pvrcnn.dataset.kitti_utils.Object3d object at 0x7fcc7d9f8a90>, 
                    <pvrcnn.dataset.kitti_utils.Object3d object at 0x7fcc7d9f8b10>, 
                    <pvrcnn.dataset.kitti_utils.Object3d object at 0x7fcc7d9f8bd0>]
            objects[i].h = data[8] # box height
            objects[i].w = data[9] # box width
            objects[i].l = data[10] # box length (in meters)
            objects[i].t = (data[11], data[12] - self.h / 2, data[13]) # location (x,y,z) in camera coord.
            objects[i].dis_to_cam = np.linalg.norm(self.t)
            objects[i].ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        """
        if cfg.DATA.DATASET == 'pandar':
            bin_name = idx
            read_calib = read_calib_pandar
        else:
            bin_name = f'{idx:06d}'
        velo_path = osp.join(cfg.DATA.ROOTDIR, 'velodyne_reduced', bin_name+'.bin')
        calib = read_calib(osp.join(cfg.DATA.ROOTDIR, 'calib', bin_name+'.txt'))
        objects = read_label(osp.join(cfg.DATA.ROOTDIR, 'label_2', bin_name+'.txt'))
        item = dict(velo_path=velo_path, calib=calib, objects=objects, idx=idx)
        self.make_simple_objects(item)
        return item

    def load_annotations(self, cfg):
        self.read_splitfile(cfg)
        try:
            self.read_cached_annotations(cfg)
        except FileNotFoundError:
            os.makedirs(cfg.DATA.CACHEDIR, exist_ok=True)
            self.create_annotations()
            self.cache_annotations(cfg)

    def _path_helper(self, folder, idx, suffix):
        return osp.join(self.cfg.DATA.ROOTDIR, folder, f'{idx:06d}.{suffix}')

    def create_annotations(self):
        self.annotations = dict()
        for idx in tqdm(self.inds, desc='Generating annotations'):
            item = dict(
                velo_path=self._path_helper('velodyne_reduced', idx, 'bin'),
                calib=read_calib(self._path_helper('calib', idx, 'txt')),
                objects=read_label(self._path_helper('label_2', idx, 'txt')), idx=idx,
            )
            self.annotations[idx] = self.make_simple_objects(item)

    def make_simple_object(self, obj, calib):
        """Converts from camera to velodyne frame."""
        xyz = calib.C2V @ np.r_[calib.R0 @ obj.t, 1]
        box = np.r_[xyz, obj.w, obj.l, obj.h, -obj.ry]
        obj = dict(box=box, class_idx=obj.class_idx)
        return obj

    def make_simple_objects(self, item):
        # item: dict(velo_path=velo_path, calib=calib, objects=objects, idx=idx)
        objects = [self.make_simple_object(obj, item['calib']) 
            for obj in item['objects']]
        item['boxes'] = np.stack([obj['box'] for obj in objects])
        item['class_idx'] = np.r_[[obj['class_idx'] for obj in objects]]
        return item

    def filter_bad_objects(self, item):
        class_idx = item['class_idx'][:, None]
        _, wlh, _ = np.split(item['boxes'], [3, 6], 1)
        keep = ((class_idx != -1) & (wlh > 0)).all(1)
        item['boxes'] = item['boxes'][keep]
        item['class_idx'] = item['class_idx'][keep]

    def filter_out_of_bounds(self, item):
        xyz, _, _ = np.split(item['boxes'], [3, 6], 1)
        lower, upper = np.split(self.cfg.GRID_BOUNDS, [3])
        keep = ((xyz >= lower) & (xyz <= upper)).all(1)
        item['boxes'] = item['boxes'][keep]
        item['class_idx'] = item['class_idx'][keep]

    def to_torch(self, item):
        item['points'] = np.float32(item['points'])
        item['boxes'] = torch.FloatTensor(item['boxes'])
        item['class_idx'] = torch.LongTensor(item['class_idx'])

    def drop_keys(self, item):
        for key in ['velo_path', 'objects', 'calib']:
            item.pop(key)

    def preprocessing(self, item):
        self.to_torch(item)

    def __getitem__(self, idx):
        try:
            return self.__getitem__core__(idx)
        except Exception as e:
            print('idx=', idx, e)
            return self.__getitem__core__(0)

    def __getitem__core__(self, idx):
        idx = idx if self.inds[idx] in self.annotations else 0
        item = deepcopy(self.annotations[self.inds[idx]])
        item['points'] = read_velo(item['velo_path'])
        self.preprocessing(item)  # excute preprocessing in KittiDatasetTrain!
        self.drop_keys(item)
        return item
        """
        self.annotations:dict, len=9102
        self.annotations['1543993993737'].keys():
        item.keys: ['velo_path', 'calib', 'objects', 'idx', 'boxes', 'class_idx']
            |
            |  ==> item['points'] = read_velo(item['velo_path'])
            V
        item.keys: ['velo_path', 'calib', 'objects', 'idx', 'boxes', 'class_idx', 'points']
            |
            |  ==> self.preprocessing(item), add: ['G_cls', 'G_reg', 'M_cls', 'M_reg']
            V
        item.keys: ['velo_path', 'calib', 'objects', 'idx', 'boxes', 'class_idx', 'points', 'G_cls', 'G_reg', 'M_cls', 'M_reg']
            |
            |  ==> self.drop_keys(item), drop: ['velo_path', 'calib', 'objects']
            V
        item.keys: ['idx', 'boxes', 'class_idx', 'points', 'G_cls', 'G_reg', 'M_cls', 'M_reg']
        """


class KittiDatasetTrain(KittiDataset):
    """TODO: This class should certainly not need access to
        anchors. Find better place to instantiate target assigner."""

    def __init__(self, cfg):
        super(KittiDatasetTrain, self).__init__(cfg, split='train')
        anchors = AnchorGenerator(cfg).anchors  # torch.Size([3, 2, 200, 176, 7])
        DatabaseBuilder(cfg, self.annotations)
        self.target_assigner = ProposalTargetAssigner(cfg, anchors)
        self.augmentation = ChainedAugmentation(cfg)

    def preprocessing(self, item):
        """Applies augmentation and assigns targets."""
        self.filter_bad_objects(item)
        points, boxes, class_idx = self.augmentation(
            item['points'], item['boxes'], item['class_idx'])
        item.update(dict(points=points, boxes=boxes, class_idx=class_idx))
        self.filter_out_of_bounds(item)
        self.to_torch(item)
        self.target_assigner(item)  # add ['G_cls', 'G_reg', 'M_cls', 'M_reg']
        """
        self.augmentation:
            before aug: (133888, 4) (22, 7) (22,)
            after aug: (149794, 4) (51, 7) (51,)
        """
