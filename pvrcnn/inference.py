import copy
import os
import os.path as osp
import numpy as np
import torch
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
from tqdm import tqdm
import time
from pvrcnn.core import cfg, Preprocessor
from pvrcnn.detector import PV_RCNN, Second
from pvrcnn.ops import nms_rotated, box_iou_rotated
from pvrcnn.core import cfg, AnchorGenerator
from viz.gen_bev import gen_bev_map, draw_bev_box

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def to_device(item):
    for key in ['points', 'features', 'coordinates', 'occupancy']:
        item[key] = item[key].cuda()
    return item

def inference(out, anchors, cfg):
    cls_map, reg_map = out['P_cls'].squeeze(0), out['P_reg'].squeeze(0)
    score_map = cls_map.sigmoid()
    top_scores, class_idx = score_map.view(cfg.NUM_CLASSES, -1).max(0)
    top_scores, anchor_idx = top_scores.topk(k=cfg.PROPOSAL.TOPK)
    class_idx = class_idx[anchor_idx]
    # import pdb;pdb.set_trace()
    top_anchors = anchors.view(cfg.NUM_CLASSES, -1, cfg.BOX_DOF)[class_idx, anchor_idx]
    top_boxes = reg_map.reshape(cfg.NUM_CLASSES, -1, cfg.BOX_DOF)[class_idx, anchor_idx]

    P_xyz, P_wlh, P_yaw = top_boxes.split([3, 3, 1], dim=1)
    A_xyz, A_wlh, A_yaw = top_anchors.split([3, 3, 1], dim=1)

    A_wl, A_h = A_wlh.split([2, 1], -1)
    A_norm = A_wl.norm(dim=-1, keepdim=True).expand(-1, 2)
    A_norm = torch.cat((A_norm, A_h), dim=-1)

    top_boxes = torch.cat((
        (P_xyz * A_norm + A_xyz),
        (torch.exp(P_wlh) * A_wlh),
        (P_yaw + A_yaw)), dim=1
    )

    nms_idx = nms_rotated(top_boxes[:, [0, 1, 3, 4, 6]], top_scores, iou_threshold=0.1)
    top_boxes = top_boxes[nms_idx]
    top_scores = top_scores[nms_idx]
    top_classes = class_idx[nms_idx]
    return top_boxes, top_scores


class Inference():
    def __init__(self,):
        self.cfg = cfg
        self.cfg.merge_from_file('../configs/second/car.yaml')
        self.preprocessor = Preprocessor(cfg)
        self.anchors = AnchorGenerator(cfg).anchors.cuda()
        self.net = PV_RCNN(cfg).cuda().eval()
        # self.net = Second(cfg).cuda().eval()
        ckpt = torch.load('./ckpts/epoch_49.pth')
        self.net.load_state_dict(ckpt['state_dict'])
        pass

    def inference_bin_to_img(self, bin_path):
        pc = np.fromfile(bin_path, np.float32).reshape(-1, 4)
        item = dict(points=[pc])
        with torch.no_grad():
            item = to_device(self.preprocessor(item))
            out = self.net(item)
            top_boxes, top_scores= inference(out, self.anchors, self.cfg)
            rgb = draw_bev_box(pc, top_boxes.cpu().numpy())
        return rgb

    def inference_bins_to_video(self, bins_dir, vid_path):
        writer = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'MJPG'), 10, (2000,1000))
        bin_names = os.listdir(bins_dir)
        bin_names.sort()
        bin_paths = [os.path.join(bins_dir, p) for p in bin_names if '.bin' in p]
        for bin_path in tqdm(bin_paths[:200]):
            rgb = self.inference_bin_to_img(bin_path).astype(np.uint8)
            writer.write(rgb)



if __name__ == '__main__':
    
    basedir = osp.join(cfg.DATA.ROOTDIR, 'velodyne_reduced/')
    bin_path = osp.join(basedir, '1544426448586.bin')
    bins_dir = '/home/kun.fan/mnt/output/lidar_baseline_20200228/20200227-154819_262'
    png_path = os.path.expanduser('~/mnt/output/1544426448586.png')
    vid_path = os.path.expanduser('~/mnt/output/test.avi')

    infer = Inference()
    rgb = infer.inference_bin_to_img(bin_path)
    cv2.imwrite(png_path, rgb)
    infer.inference_bins_to_video(bins_dir, vid_path)
