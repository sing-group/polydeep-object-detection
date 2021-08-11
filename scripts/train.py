import os
import sys
from builtins import print

import numpy as np

from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform

from gluoncv.data.transforms import experimental
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import bbox as tbbox

import mxnet as mx
import time

from gluoncv import model_zoo, utils
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from mxnet.gluon.data import DataLoader
from mxnet import autograd, gluon
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
from matplotlib import pyplot as plt
from gluoncv.utils import viz, LRSequential, LRScheduler
from mxnet import nd
from gluoncv.data.base import VisionDataset

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


class PolyDeepTrainTransformation(YOLO3DefaultTrainTransform):
    def __init__(self, width, height, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), mixup=False, **kwargs):
        super(PolyDeepTrainTransformation, self).__init__(width, height, net, mean, std, mixup, **kwargs)

    def __call__(self, src, label):
        img = src

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
            bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = timage.imresize(img, self._width, self._height, interp=interp)
        bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._target_generator is None:
            return img, bbox.astype(img.dtype)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
        if self._mixup:
            gt_mixratio = mx.nd.array(bbox[np.newaxis, :, -1:])
        else:
            gt_mixratio = None
        objectness, center_targets, scale_targets, weights, class_targets = self._target_generator(
            self._fake_x, self._feat_maps, self._anchors, self._offsets,
            gt_bboxes, gt_ids, gt_mixratio)
        return (img, objectness[0], center_targets[0], scale_targets[0], weights[0],
                class_targets[0], gt_bboxes[0])


if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# It is turn it off to disable the cuDNN algorithm exploration. Otherwise, the conv layer may generate
# non-deterministic result due to different cuDNN algorithm used.
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
# Adds an env variable to enforce determinism in the convolution operators. If set to true,
# only deterministic cuDNN algorithms will be used. If no deterministic algorithm is available, MXNet will error out.
os.environ['MXNET_ENFORCE_DETERMINISM'] = '1'


def get_train_dataset():
    train_dataset = PolyDeepDetectionDataset(root=os.path.join(dataset_dir, dataset_name),
                                             splits_root=os.path.join(os.environ['ttv_dir'], 'development'),
                                             splits=['train'])
    return train_dataset


def get_val_dataset():
    val_dataset = PolyDeepDetectionDataset(root=os.path.join(dataset_dir, dataset_name),
                                           splits_root=os.path.join(os.environ['ttv_dir'], 'development'),
                                           splits=['validation'])
    val_metric = VOCMApMetric(class_names=val_dataset.classes)
    return val_dataset, val_metric


def get_train_dataloader(net, train_dataset, batch_size, num_workers):
    # Training targets
    train_transform = PolyDeepTrainTransformation(width, height, net)
    # return stacked images, center_targets, scale_targets, gradient weights, objectness_targets, class_targets
    # additionally, return padded ground truth bboxes, so there are 7 components returned by dataloader
    batchify_fn = Tuple(*(
            [Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))

    train_loader = DataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True,
                              batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    return train_loader


def get_val_dataloader(val_dataset, batch_size, num_workers):
    val_transform = YOLO3DefaultValTransform(width, height)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = DataLoader(val_dataset.transform(val_transform), batch_size, shuffle=True,
                            batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)

    return val_loader


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
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
    map_metric, recalls, precs, scores, maxs_f1 = get(eval_metric)
    return map_metric, recalls, precs, scores, maxs_f1


def get(eval_metric):
    """Get the current evaluation result.

    Returns
    -------
    name : str
       Name of the metric.
    value : float
       Value of the evaluation.
    """
    recalls, precs, scores, maxs_f1 = update_metric(eval_metric)  # update metric at this time
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


def update_metric(eval_metric):
    """ update num_inst and sum_metric """
    aps = []
    recalls = []
    precs = []
    scores = []
    maxs_f1 = []

    recall, prec, score = recall_prec(eval_metric)
    for l, rec, pre, sco in zip(range(len(prec)), recall, prec, score):
        ap = eval_metric._average_precision(rec, pre)
        rec, pre, sco, max_f1 = recall_precision_max_f1(rec, pre, sco)
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


def recall_precision_max_f1(rec, prec, sco):
    if rec is None or prec is None:
        return np.nan

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], np.nan_to_num(prec), [0.]))
    msco = np.concatenate(([0.], np.nan_to_num(sco), [0.]))

    f1 = (mrec * mpre) / (mpre + mrec) * 2
    f1 = np.nan_to_num(f1)

    rec = mrec[np.where(f1 == np.max(f1))[0][0]]
    pre = mpre[np.where(f1 == np.max(f1))[0][0]]
    sco = msco[np.where(f1 == np.max(f1))[0][0]]

    return rec, pre, sco, np.max(f1)


def train(net, train_loader, ctx, batch_size, lr_decay_epoch, val_loader):
    net.collect_params().reset_ctx(ctx)

    if lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in lr_decay_epoch]
    lr_decay_epoch = [e - warmup_epochs for e in lr_decay_epoch]

    num_batches = len(train_dataset) // batch_size
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=learning_rate,
                    nepochs=warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler('step', base_lr=learning_rate,
                    nepochs=epochs - warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2),
    ])

    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'wd': wd, 'momentum': momentum, 'lr_scheduler': lr_scheduler},
        kvstore='local')

    # metrics
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')

    best_map = [0]

    for epoch in range(epochs):
        tic = time.time()
        for _, batch in enumerate(train_loader):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []

            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix],
                                                                      *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                autograd.backward(sum_losses)
            trainer.step(batch_size)
            obj_metrics.update(0, obj_losses)
            center_metrics.update(0, center_losses)
            scale_metrics.update(0, scale_losses)
            cls_metrics.update(0, cls_losses)

    
        if not (epoch + 1) % 1:
            obj_loss_name, obj_loss_metric = obj_metrics.get()
            center_loss_name, center_loss_metric = center_metrics.get()
            scale_loss_name, scale_loss_metric = scale_metrics.get()
            class_loss_name, class_loss_metric = cls_metrics.get()
            print(
                '[Epoch {%d}], Speed: {%.3f} samples/sec, {%s}={%.3f}, {%s}={%.3f}, '
                '{%s}={%.3f}, {%s}={%.3f}' %
                (epoch, batch_size / (time.time() - tic), obj_loss_name, obj_loss_metric,
                 center_loss_name, center_loss_metric, scale_loss_name, scale_loss_metric, class_loss_name,
                 class_loss_metric))
            print('%d,%.3f,%.3f,%.3f,%.3f,%.3f' % (
                epoch, batch_size / (time.time() - tic), obj_loss_metric,
                center_loss_metric, scale_loss_metric, class_loss_metric), file=train_results_file)

            print('Validation for epoch %d' % epoch)
            (map_name, mean_ap), recalls, precs, score_threshold, maxs_f1 = validate(net, val_loader, ctx, val_metric)
            write_val_file(epoch, map_name, mean_ap, recalls, precs, score_threshold, maxs_f1, val_results_file)
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch)


def save_params(net, best_map, current_map, epoch):
    print('Saving params in epoch %d' % epoch)
    current_map = float(current_map)
    results_dir = os.environ['results_dir']
    if current_map > best_map[0]:
        best_map[0] = current_map
        path = os.path.join(results_dir, '{:s}/{:s}.params'.format(results_dir, dataset_name))
        net.save_parameters(path)
        with open(results_dir + '/' + dataset_name + '.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))


def write_val_file(epoch, map_name, mean_ap, recalls, precs, score_threshold, maxs_f1, file_to_write):
    """
    Format to write in validation file:
    epoch;num_classes;class1_recall;class2_recall;...;class1_precision;class2_precision;...;
        class1_f1;class2_f1;...;class1_score_threshold;class2_score_threshold;...;class1;class2;...;map
    """
    val_msg = ';'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
    recall_msg = ';'.join(
        ['{}={}'.format(map_name[i] + '_recall', recalls[i]) for i in range(len(recalls))])
    prec_msg = ';'.join(
        ['{}={}'.format(map_name[i] + '_precision', precs[i]) for i in range(len(precs))])
    f1_msg = ';'.join(
        ['{}={}'.format(map_name[i] + '_f1', maxs_f1[i]) for i in range(len(maxs_f1))])
    thres_msg = ';'.join(
        ['{}={}'.format(map_name[i] + '_score_threshold', score_threshold[i]) for i in range(len(score_threshold))])
    val_msg = 'epoch=' + str(epoch) + ';num_classes=' + str(
        len(classes)) + ';' + recall_msg + ';' + prec_msg + ';' + f1_msg + ';' + thres_msg + ';' + val_msg
    print('%s' % val_msg, file=file_to_write)


# Data loader
batch_size = 8
num_workers = 0
width, height = 416, 416  # resize image to 416x416 after all data augmentation

momentum = 0.9
wd = 0.0005

learning_rate = 0.001
warmup_epochs = 0
lr_decay_epoch = (160, 180)
lr_decay = 0.1
lr_decay_period = 0

model_name = os.environ['model_name']
classes = ['polyp']
epochs = int(os.environ['epochs'])
params_dir = os.environ['params_dir']
params_name = os.environ['params_name']
seed = int(os.environ['seed'])
utils.random.seed(seed)
mx.random.seed(seed)
dataset_dir = os.environ['dataset_dir']
dataset_name = os.environ['dataset_name']

# training contexts
num_gpus = int(os.environ['num_gpus'])
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

net = model_zoo.get_model(model_name, ctx=ctx, pretrained_base=True, pretrained=True)
net.reset_class(classes, reuse_weights={'polyp': 'aeroplane'})
if params_name != "":
    print('load params')
    net.load_parameters(os.path.join(params_dir, params_name))


# train
train_dataset = get_train_dataset()
print('Training images:', len(train_dataset))
train_loader = get_train_dataloader(net, train_dataset, batch_size, num_workers)
train_results_file = open("%s/train.csv" % (os.environ['results_dir']), 'w', 1)
print('epoch,speed,obj_loss,center_loss,scale_loss,class_loss', file=train_results_file)

# validate
val_dataset, val_metric = get_val_dataset()
print('Validation images:', len(val_dataset))
val_loader = get_val_dataloader(val_dataset, batch_size, num_workers)
val_results_file = open("%s/val.csv" % (os.environ['results_dir']), 'w', 1)

train(net, train_loader, ctx, batch_size, lr_decay_epoch, val_loader)

train_results_file.close()
val_results_file.close()
