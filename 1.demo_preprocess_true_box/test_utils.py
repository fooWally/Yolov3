import numpy as np
import os, cv2
import tensorflow as tf


def get_anchors():
    '''create the anchors'''
    anchors = '10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326'
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors, np.float32).reshape(-1, 2)

def convert_to_xywh(boxes):
    """Changes the box format to center(xc,yc), width and height."""
    return tf.concat([(boxes[..., :2] + boxes[..., 2:])/2.0,
                       boxes[..., 2:] - boxes[..., :2]], axis=-1,)
def convert_to_corners(boxes):
    """Changes the box format to corner coordinates """
    return tf.concat([boxes[..., :2] - boxes[..., 2:]/2.0,
                      boxes[..., :2] + boxes[..., 2:]/2.0], axis=-1,)


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5):
        1. m is num_images to process(maybe num_batch_images)
        2. T is num_boxes that each image contains
        3. 5 entries for absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), 2 entries for wh( width and height)
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value
        xywh format represents the center of a box (xc,yc) and (width, height)
    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    print('true_boxes.shape =', true_boxes.shape)
    input_shape = np.array(input_shape, dtype='int32')
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[L] for L in range(num_layers)]
    # Build a list of empty arrays size of 13x13, 26x26 and 52x52 grids
    y_true = [np.zeros((m, grid_shapes[L][0],grid_shapes[L][1], len(anchor_mask[L]), 5+num_classes),
                dtype='float32') for L in range(num_layers)]

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.0
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0
    #-----------------------------------------------------------------
    # Find the position of best_anchor, which masks the ground true box the best 
    # and store the scaled true_boxes in that position (j,i) in y_true for each grid
    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins  = -box_maxes

        intersect_mins  = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            # L=0 for 13x13, L=1 for 26x26, L=2 for 52x52
            for L in range(num_layers):
                if n in anchor_mask[L]:
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[L][1]).astype('int32')
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[L][0]).astype('int32')
                    k = anchor_mask[L].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[L][b, j, i, k, 0:4] = true_boxes[b, t, 0:4] # scaled box in format of [xc,yc,w,h]
                    y_true[L][b, j, i, k,   4] = 1  # for objectness. 0 for background
                    y_true[L][b, j, i, k, 5+c] = 1  # one-hot encoding for class id
    #--------------------------------------------------
    # RETURN iou, best_anchor for developing code ONLY
    return y_true, iou, best_anchor


def image_preporcess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def parse_annotation(annotation,TRAIN_IMG_SIZE = 416):
    
    line = annotation.split()
    image_path = line[0]
    box = line[1:]
    #print('box =', box)
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " %image_path)
    image = cv2.imread(image_path)
    #print('image.shape =', image.shape)
    bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, bboxes = image_preporcess(np.copy(image), [TRAIN_IMG_SIZE, TRAIN_IMG_SIZE], np.copy(bboxes))
    return image, bboxes

def load_annotations(annot_path):
    with open(annot_path, 'r') as f:
        txt = f.readlines()[1:]
        annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    return annotations
