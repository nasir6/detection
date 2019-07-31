import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class TXTDataset(CustomDataset):

    def __init__(self, min_size=None, **kwargs):
        super(TXTDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        self.cat_ids = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}


    def str2int(self, a):
        return [int(x) for x in a]

    def extract_boxes(self, fname):
        with open(fname) as f:
            content = f.readlines()
            f.close()
            content = [x.strip().split(' ') for x in content]
            return content

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        self.img_ids = img_ids
        for img_id in img_ids:
            filename = 'JPEGImages/{}.jpg'.format(img_id)
            txt_path = osp.join(self.img_prefix, 'Annotations',
                                '{}{}.txt'.format(img_id, self.anno_file_postfix))
            
            info = self.extract_boxes(txt_path)

            width = int(info[0][1])
            height = int(info[0][2])
            img_infos.append(
                dict(id=img_id, filename=filename, width=width, height=height))
        
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        txt_path = osp.join(self.img_prefix, 'Annotations',
                            '{}{}.txt'.format(img_id, self.anno_file_postfix))
        
        anno_data = self.extract_boxes(txt_path)
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in anno_data:
            name = obj[0]
            label = self.cat2label[name]
            difficult = int(obj[3])
            bbox = self.str2int(obj[-4:])
            
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
