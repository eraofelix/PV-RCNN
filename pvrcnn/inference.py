import os.path as osp
import numpy as np
import torch

from pvrcnn.core import cfg, Preprocessor
from pvrcnn.detector import PV_RCNN


def to_device(item):
    for key in ['points', 'features', 'coordinates', 'occupancy']:
        item[key] = item[key].cuda()
    return item


def main():
    cfg.merge_from_file('../configs/car.yaml')
    preprocessor = Preprocessor(cfg)
    net = PV_RCNN(cfg).cuda().eval()
    ckpt = torch.load('./ckpts/old/epoch2_2.pth')
    net.load_state_dict(ckpt['state_dict'])
    basedir = osp.join(cfg.DATA.ROOTDIR, 'velodyne_reduced/')
    item = dict(points=[
        np.fromfile(osp.join(basedir, '1544505305505.bin'), np.float32).reshape(-1, 4),
    ])
    with torch.no_grad():
        item = to_device(preprocessor(item))
        out = net.proposal(item)


if __name__ == '__main__':
    main()
