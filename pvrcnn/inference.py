import os
import os.path as osp
import numpy as np
import torch
from pvrcnn.core import cfg, Preprocessor
from pvrcnn.detector import PV_RCNN, Second
from pvrcnn.ops import nms_rotated, box_iou_rotated
from pvrcnn.core import cfg, AnchorGenerator
import copy
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

def main():
    cfg.merge_from_file('../configs/second/car.yaml')
    preprocessor = Preprocessor(cfg)
    anchors = AnchorGenerator(cfg).anchors.cuda()
    net = PV_RCNN(cfg).cuda().eval()
    # net = Second(cfg).cuda().eval()
    ckpt = torch.load('./ckpts/epoch_0.pth')
    net.load_state_dict(ckpt['state_dict'])
    basedir = osp.join(cfg.DATA.ROOTDIR, 'velodyne_reduced/')
    item = dict(points=[
        np.fromfile(osp.join(basedir, '1544426448586.bin'), np.float32).reshape(-1, 4),
    ])
    with torch.no_grad():
        item = to_device(preprocessor(item))
        out = net(item)
        top_boxes, top_scores= inference(out, anchors, cfg)
        print('top_boxes:', top_boxes)
        print('top_scores', top_scores)


if __name__ == '__main__':
    main()
