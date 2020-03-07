import numpy as np
import cv2
import os


def load_pc(f):
    b = np.fromfile(f, dtype=np.float32)
    return b.reshape((-1, 4))[:, :3]


def gen_bev_map(pc, lr=100, bf=200, res=0.1):
    """generate bew image from 3d pointcloud

    Args:
        pc (np.ndarray): [description]
        lr (left-right distance, optional): [description]. Defaults to 100.
        bf (back-froward distance, optional): [description]. Defaults to 200.
        res (float, optional): resolution. Defaults to 0.05.

    Returns:
        im: np.ndarray, shape(lr, bf)
        rgb: np.ndarray, shape(lr, bf, 3)
        cropped_cloud: 
    """
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    # filter point cloud
    f_filt = np.logical_and((x > -bf/2), (x < bf/2))
    s_filt = np.logical_and((y > -lr/2), (y < lr/2))
    filt = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filt).flatten()
    x = x[indices]
    y = y[indices]
    z = z[indices]

    # convert coordinates to
    x_img = (-y/res).astype(np.int32)
    y_img = (-x/res).astype(np.int32)
    # shifting image, make min pixel is 0,0
    x_img += int(np.floor(lr/2/res))
    y_img += int(np.ceil(bf/2/res))

    # crop y to make it not bigger than 255
    height_range = (-3, 2)
    pixel_values = np.clip(a=z, a_min=height_range[0], a_max=height_range[1])

    def scale_to_255(a, min, max, dtype=np.uint8):
        return (((a - min) / float(max - min)) * 255).astype(dtype)
    pixel_values = scale_to_255(
        pixel_values, min=height_range[0], max=height_range[1])

    # according to width and height generate image
    w = int(lr/res)
    h = int(bf/res)
    im = np.zeros([h, w], dtype=np.uint8)
    im[y_img, x_img] = pixel_values
    im = im[::-1].transpose()
    rgb = np.stack([np.zeros(im.shape), np.zeros(im.shape), im], -1)
    cropped_cloud = np.vstack([x, y, z]).transpose()

    return im, rgb, cropped_cloud


def xyzwlht_to_polygonpoints(xyzwlh):
    """xyzwlht_to_polygonpoints [summary]

    Args:
        xyzwlht (np.ndarray): [7, ]: x,y,z,w,l,h,theta
        polygonpoints (np.ndarray): [4,2]: [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]

        x0,y0 ----x1,y1
        |          |
        |          |
        |          |
        x3,y3-----x2,y2
    """
    x = xyzwlh[0]
    y = xyzwlh[1]
    z = xyzwlh[2]
    ObsTheta = xyzwlh[6] + np.pi / 2
    Length = xyzwlh[4]
    Width = xyzwlh[3]
    Height = xyzwlh[5]

    x0 = x - Width/2.0 * np.sin(ObsTheta) - Length / 2.0 * np.cos(ObsTheta)
    y0 = y + Width/2.0 * np.cos(ObsTheta) - Length / 2.0 * np.sin(ObsTheta)

    x1 = x + Width/2.0 * np.sin(ObsTheta) - Length / 2.0 * np.cos(ObsTheta)
    y1 = y - Width/2.0 * np.cos(ObsTheta) - Length / 2.0 * np.sin(ObsTheta)

    x2 = x + Width/2.0 * np.sin(ObsTheta) + Length / 2.0 * np.cos(ObsTheta)
    y2 = y - Width/2.0 * np.cos(ObsTheta) + Length / 2.0 * np.sin(ObsTheta)

    x3 = x - Width/2.0 * np.sin(ObsTheta) + Length / 2.0 * np.cos(ObsTheta)
    y3 = y + Width/2.0 * np.cos(ObsTheta) + Length / 2.0 * np.sin(ObsTheta)

    return np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])


def draw_bev_box(pc, boxes):
    """draw_bev_box [summary]

    Args:
        pc (np.ndarray): [N, 3]
        boxes (np.ndarray): [box_num, 7]
    """
    im, rgb, _ = gen_bev_map(pc)
    h, w = im.shape
    scale = h / 100
    for box in boxes:
        points = (xyzwlht_to_polygonpoints(box)*scale).astype(np.int32)
        points[:, 0] = points[:, 0] + w//2
        points[:, 1] = h//2 - points[:, 1]
        cv2.polylines(rgb, np.int32([points]), 1, (0,255,0))

    return rgb


if __name__ == '__main__':

    a = '/Users/felix/tmp/1544426448586.bin'
    points = load_pc(a)
    boxes = np.array([[10.8142,  4.3419, -1.5002,  1.7520,  4.2997,  1.5199,  1.6309],
                      [17.6510, -2.2238, -1.6200,  1.7596,
                          4.2559,  1.5404,  1.6067],
                      [19.1842,  4.1860, -1.5994,  1.7713,
                          4.2742,  1.5262,  1.5747],
                      [20.6845, 26.0792, -1.8360,  1.7538,
                          4.3198,  1.5311,  1.6847],
                      [7.6023,  0.2244, -1.4496,  1.7894,
                          4.3437,  1.5543,  1.6053],
                      [2.9149,  4.2401, -1.4803,  1.7520,
                          4.7347,  1.6105,  1.6369],
                      [27.1094,  8.8551, -1.7229,  1.7546,
                          4.2266,  1.5490,  1.5938],
                      [49.2019,  5.6506, -2.0619,  1.7468,
                          4.3082,  1.5326,  1.6352],
                      [2.7610,  8.0643, -1.2577,  1.6359,
                          4.1076,  1.5095,  1.6099],
                      [38.1095, -1.5373, -1.8322,  1.8133,
                          4.3309,  1.5765,  1.6383],
                      [11.9913,  7.9866, -1.5131,  1.7195,  4.1635,  1.5327,  1.5961]])
    draw_bev_box(points, boxes)
