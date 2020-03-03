import os
import os.path as osp
import numpy as np
import torch
from pvrcnn.core import cfg, Preprocessor
from pvrcnn.detector import PV_RCNN
from pvrcnn.ops import nms_rotated, box_iou_rotated
from pvrcnn.core import cfg, AnchorGenerator
import copy
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def to_device(item):
    for key in ['points', 'features', 'coordinates', 'occupancy']:
        item[key] = item[key].cuda()
    return item

def get_topk(out, anchors, cfg):
    # tensor2numpy
    cls_map = out['P_cls'].cpu().numpy().squeeze(0)  # (2, 2, 200, 176) (NUM_CLASSES+1, NUM_YAW, ny, nx)
    reg_map = out['P_reg'].cpu().numpy().squeeze(0)  # (1, 2, 200, 176, 7)
    anchors = anchors.cpu().numpy()       # (1, 2, 200, 176, 7)
    score_map = 1/(1+np.exp(-cls_map))  # along cls dim!!!

    top_scores = score_map.reshape([cfg.NUM_CLASSES+1, -1]).max(0)
    top_scores_copy = copy.deepcopy(top_scores)
    top_scores.sort()
    top_scores = top_scores[-cfg.PROPOSAL.TOPK:][::-1]
    anchor_idx = top_scores_copy.argsort()[-cfg.PROPOSAL.TOPK:][::-1]

    top_anchors = anchors.reshape([cfg.NUM_CLASSES, -1, cfg.BOX_DOF])
    top_anchors = top_anchors[:, anchor_idx, :]

    top_boxes = reg_map.reshape([cfg.NUM_CLASSES, -1, cfg.BOX_DOF])
    top_boxes = top_boxes[:, anchor_idx, :]

    
    P_xyz, P_wlh, P_yaw = top_boxes[:, :, :3], top_boxes[:, :, 3:6], top_boxes[:, :, 6:]
    A_xyz, A_wlh, A_yaw = top_anchors[:, :, :3], top_anchors[:, :, 3:6], top_anchors[:, :, 6:]
    
    A_wl, A_h = A_wlh[:, :, :2], A_wlh[:, :, 2:]
    A_norm = np.linalg.norm(A_wl, axis=-1, keepdims=True)  # (1,100,1)
    A_norm = np.concatenate([A_norm, A_norm, A_h], axis=-1)  # (1,100,2)
    top_boxes = np.concatenate([P_xyz * A_norm + A_xyz,
                                np.exp(P_wlh) * A_wlh, 
                                P_yaw + A_yaw], axis=-1)[0]
    # numpy2tensor
    top_scores = torch.from_numpy(top_scores-np.zeros_like(top_scores)).float().cuda()
    top_boxes = torch.from_numpy(top_boxes).float().cuda()

    # [0, 1, 3, 4, 6] ==>[x, y, w, l, yaw]
    nms_idx = nms_rotated(top_boxes[:, [0, 1, 3, 4, 6]], top_scores, iou_threshold=0.2)
    top_boxes = top_boxes[nms_idx]
    top_scores = top_scores[nms_idx]
    return top_boxes, top_scores

def inference_my(out, anchors, cfg):  # anchors: torch.Size([1, 2, 200, 176, 7])
    cls_map, reg_map = out['P_cls'].squeeze(0), out['P_reg'].squeeze(0)
    # cls_map.Size([2, 2, 200, 176])
    # reg_map: torch.Size([1, 2, 200, 176, 7])
    score_map = cls_map.sigmoid()  
    top_scores, class_idx = score_map.view(cfg.NUM_CLASSES, -1).max(0)
    top_scores, anchor_idx = top_scores.topk(k=cfg.PROPOSAL.TOPK)  # 100
    class_idx = class_idx[anchor_idx]
    top_anchors = anchors.view(cfg.NUM_CLASSES, -1, cfg.BOX_DOF)  # torch.Size([1, 70400, 7])
    top_anchors = torch.index_select(top_anchors, 1, anchor_idx) # torch.Size([1, 100, 7])
    top_boxes = reg_map.reshape(cfg.NUM_CLASSES, -1, cfg.BOX_DOF) # torch.Size([1, 70400, 7])
    top_boxes = torch.index_select(top_boxes, 1, anchor_idx) # torch.Size([1, 100, 7])
    P_xyz, P_wlh, P_yaw = top_boxes.split([3, 3, 1], dim=-1)
    A_xyz, A_wlh, A_yaw = top_anchors.split([3, 3, 1], dim=-1)

    A_wl, A_h = A_wlh.split([2, 1], -1) # torch.Size([1, 100, 2]), torch.Size([1, 100, 1])

    A_norm = A_wl#.norm(dim=-1, keepdim=True)
    A_norm = A_norm.expand(-1, 2)
    A_norm = torch.cat((A_norm, A_h), dim=-1)

    top_boxes = torch.cat((  # (100, 7), cfg.PROPOSAL.TOPK=100
        (P_xyz * A_norm + A_xyz),
        (torch.exp(P_wlh) * A_wlh),
        (P_yaw + A_yaw)), dim=1
    )
    
    # [0, 1, 3, 4, 6] ==>[x, y, w, l, yaw]
    nms_idx = nms_rotated(top_boxes[:, [0, 1, 3, 4, 6]], top_scores, iou_threshold=0.01)
    top_boxes = top_boxes[nms_idx]
    top_scores = top_scores[nms_idx]
    
    return top_boxes, top_scores

def inference(out, anchors, cfg):
    cls_map, reg_map = out['P_cls'].squeeze(0), out['P_reg'].squeeze(0)
    score_map = cls_map.sigmoid()
    top_scores, class_idx = score_map.view(cfg.NUM_CLASSES, -1).max(0)
    top_scores, anchor_idx = top_scores.topk(k=cfg.PROPOSAL.TOPK)
    class_idx = class_idx[anchor_idx]
    import pdb;pdb.set_trace()
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

    nms_idx = nms_rotated(top_boxes[:, [0, 1, 3, 4, 6]], top_scores, iou_threshold=0.01)
    top_boxes = top_boxes[nms_idx]
    top_scores = top_scores[nms_idx]
    top_classes = class_idx[nms_idx]
    return top_boxes, top_scores

def main():
    cfg.merge_from_file('../configs/car.yaml')
    preprocessor = Preprocessor(cfg)
    anchors = AnchorGenerator(cfg).anchors.cuda()
    net = PV_RCNN(cfg).cuda().eval()
    ckpt = torch.load('./ckpts/epoch_30.pth')
    net.load_state_dict(ckpt['state_dict'])
    basedir = osp.join(cfg.DATA.ROOTDIR, 'velodyne_reduced/')
    item = dict(points=[
        np.fromfile(osp.join(basedir, '1544426448586.bin'), np.float32).reshape(-1, 4),
    ])
    with torch.no_grad():
        item = to_device(preprocessor(item))
        out = net.proposal(item)
        top_boxes, top_scores= inference(out, anchors, cfg)
        print('top_boxes:', top_boxes)
        print('top_scores', top_scores)


if __name__ == '__main__':
    main()
