import os
import re
import sys
from builtins import print

import numpy as np
import mxnet as mx

from gluoncv import model_zoo
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils import viz
from mxnet.gluon.data import DataLoader
from mxnet import gluon
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
from gluoncv.data.base import VisionDataset
from matplotlib import pyplot as plt

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class PolyDeepDetectionDataset(VisionDataset):
    """PolyDeep Dataset for polyp detection. Inspired in VOC structure, but there is no 'year' and splits are made
        outside the dataset root, so we add a splits_root parameter. In addition, object locations are 0-based.
        """

    CLASSES = ['polyp']

    def __init__(self, root, splits_root, splits, transform=None, index_map=None, preload_label=True):
        super(PolyDeepDetectionDataset, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits_root = splits_root
        self._splits = splits
        self._items = self._load_items(splits)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        try:
            self._validate_class_names(self.CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for split in splits:
            lf = os.path.join(self._splits_root)
            if isinstance(split, str):
                file = split + '.txt'
                lf = os.path.join(lf, split, file)
            else:
                split = list(split)
                split.append(split[-1] + '.txt')
                split = tuple(split)
                for subsplit in split:
                    lf = os.path.join(lf, subsplit)
            with open(lf, 'r') as f:
                ids += [(self._root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            try:
                difficult = int(obj.find('difficult').text)
            except ValueError:
                difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        return [self._load_label(idx) for idx in range(len(self))]


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# It is turn it off to disable the cuDNN algorithm exploration. Otherwise, the conv layer may generate
# non-deterministic result due to different cuDNN algorithm used.
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
# Adds an env variable to enforce determinism in the convolution operators. If set to true,
# only deterministic cuDNN algorithms will be used. If no deterministic algorithm is available, MXNet will error out.
os.environ['MXNET_ENFORCE_DETERMINISM'] = '1'


def get_test_dataset():
    test_dataset = PolyDeepDetectionDataset(root=os.path.join(dataset_dir, dataset_name),
                                            splits_root=os.environ['ttv_dir'],
                                            splits=['test'])
    test_metric = VOCMApMetric(class_names=test_dataset.classes)
    return test_dataset, test_metric


def get_test_dataloader(test_dataset, batch_size, num_workers):
    test_transform = YOLO3DefaultValTransform(width, height)
    return DataLoader(test_dataset.transform(test_transform), batch_size, shuffle=True,
                      batchify_fn=Tuple(Stack(), Pad(pad_val=-1)), last_batch='keep', num_workers=num_workers)


def test(net, val_data, ctx, eval_metric, threshold):
    """Test on test dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    num_images = 0
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            num_images = num_images + 1
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)

    print('Number of images:', num_images)
    map_metric, recalls, precs, scores, maxs_f1 = get(eval_metric, threshold)
    return map_metric, recalls, precs, scores, maxs_f1


def get(eval_metric, threshold):
    """Get the current evaluation result.

    Returns
    -------
    name : str
       Name of the metric.
    value : float
       Value of the evaluation.
    """
    recalls, precs, scores, maxs_f1 = update_metric(eval_metric, threshold)  # update metric at this time
    if eval_metric.num is None:
        if eval_metric.num_inst == 0:
            return (eval_metric.name, float('nan'))
        else:
            return (eval_metric.name, eval_metric.sum_metric / eval_metric.num_inst)
    else:
        names = ['%s' % (eval_metric.name[i]) for i in range(eval_metric.num)]
        values = [x / y if y != 0 else float('nan') \
                  for x, y in zip(eval_metric.sum_metric, eval_metric.num_inst)]
        return (names, values), recalls, precs, scores, maxs_f1


def update_metric(eval_metric, threshold):
    """ update num_inst and sum_metric """
    aps = []
    recalls = []
    precs = []
    scores = []
    maxs_f1 = []

    recall, prec, score = recall_prec(eval_metric)
    for l, rec, pre, sco in zip(range(len(prec)), recall, prec, score):
        ap = eval_metric._average_precision(rec, pre)
        rec, pre, sco, max_f1 = recall_precision_max_f1(rec, pre, sco, threshold)
        recalls.append(rec)
        precs.append(pre)
        aps.append(ap)
        scores.append(sco)
        maxs_f1.append(max_f1)
        if eval_metric.num is not None and l < (eval_metric.num - 1):
            eval_metric.sum_metric[l] = ap
            eval_metric.num_inst[l] = 1
    if eval_metric.num is None:
        eval_metric.num_inst = 1
        eval_metric.sum_metric = np.nanmean(aps)
    else:
        eval_metric.num_inst[-1] = 1
        eval_metric.sum_metric[-1] = np.nanmean(aps)
    return recalls, precs, scores, maxs_f1


def recall_prec(eval_metric):
    """ get recall and precision from internal records """
    n_fg_class = max(eval_metric._n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    score = [None] * n_fg_class

    for l in eval_metric._n_pos.keys():
        score_l = np.array(eval_metric._score[l])
        match_l = np.array(eval_metric._match[l], dtype=np.int32)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        with np.errstate(divide='ignore', invalid='ignore'):
            prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if eval_metric._n_pos[l] > 0:
            rec[l] = tp / eval_metric._n_pos[l]

        score[l] = score_l[order]

    return rec, prec, score


def recall_precision_max_f1(rec, prec, sco, threshold):
    if rec is None or prec is None:
        return np.nan

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], np.nan_to_num(prec), [0.]))
    msco = np.concatenate(([0.], np.nan_to_num(sco), [0.]))

    f1 = (mrec * mpre) / (mpre + mrec) * 2
    f1 = np.nan_to_num(f1)

    # take the first line where the score is less than threshold in order to obtain recall and precision for that
    # score
    position = np.where(np.nan_to_num(sco) <= float(threshold))[0][0]
    if msco[position + 1] != threshold:
        position = position - 1

    rec = mrec[position + 1]
    pre = mpre[position + 1]
    sco = msco[position + 1]

    return rec, pre, sco, f1[position + 1]


def write_test_file(map_name, mean_ap, recalls, precs, score_threshold, maxs_f1, threshold, file_to_write):
    """
    Format to write in test file:
    epoch;num_classes;class1_recall;class2_recall;...;class1_precision;class2_precision;...;
        class1_f1;class2_f1;...;class1_score_threshold;class2_score_threshold;...;class1;class2;...;map
    """
    test_msg = ';'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    recall_msg = ';'.join(
        ['{}={}'.format(map_name[i] + '_recall', recalls[i]) for i in range(len(recalls))])
    prec_msg = ';'.join(
        ['{}={}'.format(map_name[i] + '_precision', precs[i]) for i in range(len(precs))])
    f1_msg = ';'.join(
        ['{}={}'.format(map_name[i] + '_f1', maxs_f1[i]) for i in range(len(maxs_f1))])
    thres_msg = ';'.join(
        ['{}={}'.format(map_name[i] + '_score_threshold', score_threshold[i]) for i in range(len(score_threshold))])

    test_msg = 'num_classes=' + str(
        len(classes)) + ';' + recall_msg + ';' + prec_msg + ';' + f1_msg + ';' + thres_msg + ';threshold=' + str(
        threshold) + ';' + test_msg
    print('%s' % test_msg, file=file_to_write)


# Data loader
batch_size = 8
num_workers = 0
width, height = 416, 416  # resize image to 416x416 after all data augmentation

model_name = os.environ['model_name']
classes = ['polyp']
dataset_dir = os.environ['dataset_dir']
dataset_name = os.environ['dataset_name']
results_dir = os.environ['results_dir']

# training contexts
num_gpus = int(os.environ['num_gpus'])
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

net = model_zoo.get_model(model_name, ctx=ctx, pretrained_base=True, pretrained=True)
net.reset_class(classes, reuse_weights={'polyp': 'aeroplane'})

print('loading network')
net.load_parameters(os.path.join(os.environ['results_dir'], os.environ['neuronal_network']))
print(os.environ['neuronal_network'] + " with threshold " + os.environ['threshold_cnn'])
threshold = os.environ['threshold_cnn']

# test
test_dataset, test_metric = get_test_dataset()
print('Test images:', len(test_dataset))
test_loader = get_test_dataloader(test_dataset, batch_size, num_workers)
test_results_file = open("%s/test.csv" % results_dir, 'w', 1)

(map_name_train, mean_ap_train), recalls, precs, score_threshold, maxs_f1 = test(net, test_loader, ctx, test_metric,
                                                                                 threshold)
write_test_file(map_name_train, mean_ap_train, recalls, precs, score_threshold, maxs_f1, threshold, test_results_file)

test_results_file.close()
