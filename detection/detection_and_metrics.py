import numpy as np
import torch
import torch.nn as nn 


# ============================== 1 Classifier model ============================

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    # your code here \/
    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(16),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(2),  

        nn.Flatten(start_dim=1, end_dim=-1),
        nn.LazyLinear(64),
        nn.ReLU(),
        nn.LazyLinear(2),
    )
    return model
    # your code here /\

def fit_cls_model(X, y):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """
    # your code here \/
    model = get_cls_model((40, 100, 1))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs = 100
    # train model
    for i in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        #print(y_pred.max(axis=1) == y, y_pred)
        acc = (y_pred.argmax(axis=1) == y).sum() / len(y)
        print(f"Epoch {i}, Loss: {loss.item()}, Accuracy: {acc}")
        loss.backward()
        optimizer.step()
    #torch.save(model, "classifier_model.pth")
    return model
    # your code here /\


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    # your code here \/
    after_flatten = False
    conv_model = []
    for layer in cls_model.children():
        if isinstance(layer, nn.Flatten):
            after_flatten = True
        elif isinstance(layer, nn.Linear):
            W, b = layer.parameters()
            if after_flatten:
                conv_layer = nn.Conv2d(32, W.shape[0], kernel_size=(20, 50), stride=(1, 3))
                after_flatten = False
            else:
                conv_layer = nn.Conv2d(W.shape[1], W.shape[0], kernel_size=1, stride=1)
            conv_layer.weight = nn.Parameter(W.reshape(conv_layer.weight.shape))
            conv_layer.bias = nn.Parameter(b)
            conv_model.append(conv_layer)
        else:
            conv_model.append(layer)

    detection_model = nn.Sequential(*conv_model)
    return detection_model
    # your code here /\

# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    images = []
    filenames = []
    for filename, image in dictionary_of_images.items():
        padded_image = torch.zeros((1, 220, 370))
        h, w = image.shape[:2]
        padded_image[..., :h, :w] = torch.from_numpy(image)
        images.append(padded_image)
        filenames.append(filename)
    
    images = torch.stack(images, 0)
    detection_model.eval()
    feature_map = []
    batch_size = 64
    with torch.no_grad():
        for i in range(len(images) // batch_size + 1):
            outputs = detection_model(images[i*batch_size : (i+1)*batch_size])
            feature_map.append(torch.nn.Softmax(dim=1)(outputs))
    feature_map = torch.cat(feature_map)
    detections = {}
    for filename, detection in zip(filenames, feature_map):
        rows, cols = torch.where(detection[1] > 0.99)
        detections[filename] = []
        for row, col in zip(rows, cols):
            if row*2 + 40 < dictionary_of_images[filename].shape[0] and col*2 + 100 < dictionary_of_images[filename].shape[1]:
                detections[filename].append((row*2, col*2, 40, 100, detection[1, row, col]))
    return detections
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    row, col, n_rows, n_cols = first_bbox
    x1_l, y1_l, x1_r, y1_r = row, col, row+n_rows, col+n_cols
    row, col, n_rows, n_cols = second_bbox
    x2_l, y2_l, x2_r, y2_r = row, col, row+n_rows, col+n_cols
    x_l, x_r = max(x1_l, x2_l), min(x1_r, x2_r)
    y_l, y_r = max(y1_l, y2_l), min(y1_r, y2_r)
    if x_l >= x_r or y_l >= y_r:
        return 0
    intersect = (y_r - y_l) * (x_r - x_l)
    union = (y1_r - y1_l)*(x1_r - x1_l) + (y2_r - y2_l)*(x2_r - x2_l) - intersect
    return intersect / union
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    # your code here \/
    num_gt = 0
    tp_all, fp_all = [], []
    for filename, detections in pred_bboxes.items():
        tp, fp = [], []
        gt = gt_bboxes[filename]
        num_gt += len(gt)
        for detection in sorted(detections, key = lambda x: x[4], reverse=True):
            prev_iou, prev_true_box = 0, []
            for true_box in gt:
                iou = calc_iou(detection[:4], true_box)
                if iou > 0.5 and iou > prev_iou:
                    prev_iou = iou
                    prev_true_box = true_box
            if len(prev_true_box) > 0:
                tp.append(detection[4])
                gt.remove(prev_true_box)
            else:
                fp.append(detection[4])
        tp_all += tp
        fp_all += fp
    y_pred = np.array(sorted(tp_all+fp_all))
    tp_all = np.array(sorted(tp_all))
    precision_recall_curve = []
    for c in y_pred:
        pred = (y_pred >= c).sum()
        pred_tp = (tp_all >= c).sum()
        precision = pred_tp / pred
        recall = pred_tp / num_gt
        precision_recall_curve.append((precision, recall, c))
    precision_recall_curve.append((1, 0, None))
    auc = 0
    for i in range(len(precision_recall_curve) - 1):
        recall = precision_recall_curve[i][1] - precision_recall_curve[i+1][1]
        precision = (precision_recall_curve[i][0] + precision_recall_curve[i+1][0]) / 2
        auc += precision * recall
    print(auc)
    return auc
    # your code here /\


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.7):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    for filename, detections in detections_dictionary.items():
        detections_sort = sorted(detections, key=lambda item: item[4], reverse=True)
        remove_detections = set()
        clean_detections = []
        for i in range(len(detections_sort)):
            if i in remove_detections:
                continue
            for j in range(i+1, len(detections_sort)):
                iou = calc_iou(detections_sort[i][:-1], detections_sort[j][:-1])
                if iou > iou_thr:
                    remove_detections.add(j)
            clean_detections.append(detections_sort[i])
        detections_dictionary[filename] = clean_detections
    return detections_dictionary
    # your code here /\
