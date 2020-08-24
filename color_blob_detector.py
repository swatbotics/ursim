######################################################################
#
# sample_colors.py
# 
# Written for ENGR 028/CPSC 082: Mobile Robotics, Summer 2020
# Copyright (C) Matt Zucker 2020
#
######################################################################

import cv2
import numpy
import os
import json
import sys

def numpy_from_list(l):
    return numpy.array(l, dtype=numpy.uint8)

def list_from_numpy(a):
    if len(a.shape) > 1:
        return [ list_from_numpy(ai) for ai in a ]
    else:
        return [ int(ai) for ai in a ]

def replace_recursive(d, cls, fn):

    if isinstance(d, cls):
        try:
            return fn(d)
        except:
            pass
    
    if isinstance(d, dict):
        rval = dict()
        for key, value in d.items():
            key = replace_recursive(key, cls, fn)
            value = replace_recursive(value, cls, fn)
            rval[key] = value
        return rval
    elif isinstance(d, list):
        rval = list()
        for item in d:
            rval.append(replace_recursive(item, cls, fn))
        return rval
    else:
        return d

def numpy_from_json(d):
    return replace_recursive(d, list, numpy_from_list)

def json_from_numpy(d):
    return replace_recursive(d, numpy.ndarray, list_from_numpy)

def union(boxa, boxb):
    if boxa is None:
        return boxb.copy()
    elif boxb is None:
        return boxa.copy()
    else:
        return numpy.array([
            numpy.minimum(boxa[0], boxb[0]),
            numpy.maximum(boxa[1], boxb[1])
        ], dtype=boxa.dtype)


def intersection(boxa, boxb):
    if boxa is None:
        return None
    elif boxb is None:
        return None
    else:
        a = numpy.array([
            numpy.maximum(boxa[0], boxb[0]),
            numpy.minimum(boxa[1], boxb[1])
        ], dtype=boxa.dtype)
        if numpy.any(a[0] > a[1]):
            return None
        else:
            return a

def json_encode(obj, indent=0):
    istr = '  ' * indent
    iistr = istr + '  '
    if isinstance(obj, str):
        return '"{}"'.format(obj)
    elif isinstance(obj, int):
        return '{:>3d}'.format(obj)
    elif isinstance(obj, list):
        enc_items = [ json_encode(item, indent+1) for item in obj ]
        enc_lengths = [ len(i) for i in enc_items ]
        total_length = sum(enc_lengths) + 4 + 2*(len(enc_items)-1)
        if total_length < 60 and not any([('\n' in i) for i in enc_items]):
            return '[ ' + ', '.join(enc_items) + ' ]'
        else:
            return '[\n' + iistr + (',\n' + iistr).join(enc_items) + '\n' + istr + ']'
    elif isinstance(obj, dict):
        enc_items = [ (json_encode(key, indent+1), json_encode(value, indent+1)) for key, value in obj.items() ]
        max_key = 0
        for k, v in enc_items:
            if max_key is None or '\n' in k or '\n' in v:
                max_key = None
            else:
                max_key = max(max_key, len(k))
        if max_key is not None:
            enc_items = [ k + ': ' + ' '*(max_key-len(k)) + v for k, v in enc_items ]
        else:
            enc_items = [ k + ': ' + v for k, v in enc_items ]
        return '{\n' + iistr + (',\n' + iistr).join(enc_items) + '\n' + istr + '}'
    else:
        raise RuntimeError('nope')


def get_mask(bbox, image):
 
    smin, smax = bbox
    
    greater = (image >= smin).min(axis=2)
    less = (image <= smax).min(axis=2)
    
    return (greater & less)

def label_image(all_bboxes, image, labels=None):

    nlabels = len(all_bboxes)
    assert all_bboxes.shape == (nlabels, 2, 3)

    h, w = image.shape[:2]
    assert image.shape == (h, w, 3)

    assert nlabels <= 8

    init_mask = (1 << nlabels) - 1

    if labels is None:
        labels = numpy.empty((h, w), dtype=numpy.uint8)
    else:
        assert labels.shape == (h, w) and labels.dtype == numpy.uint8
        
    labels[:] = init_mask

    for label_index in range(nlabels):
        label_mask = ~numpy.uint8(1 << label_index)
        for channel in range(3):
            ichan = image[:, :, channel]
            lo = all_bboxes[label_index, 0, channel]
            hi = all_bboxes[label_index, 1, channel]
            labels[ichan < lo] &= label_mask
            labels[ichan > hi] &= label_mask

    return labels

class ColorBlobDetector:

    def __init__(self, config_filename=None, mode='bgr'):

        assert mode in ['bgr', 'rgb']

        self.mode = mode

        if mode == 'bgr':
            self.to_ycrcb = cv2.COLOR_BGR2YCrCb
            self.from_ycrcb = cv2.COLOR_YCrCb2BGR
        else:
            self.to_ycrcb = cv2.COLOR_RGB2YCrCb
            self.from_ycrcb = cv2.COLOR_YCrCb2RGB

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if config_filename is None:
            self.json_filename = os.path.join(self.script_dir, 'color_definitions.json')
        else:
            self.json_filename = config_filename

        with open(self.json_filename, 'r') as istr:
            cdata = numpy_from_json(json.load(istr))

        colors = cdata['colors']

        assert isinstance(colors, list)
        ckeys = set(['name', 'bounds_ycrcb'])

        self.color_names = []
        self.all_bboxes = []
        
        for color in colors:
            assert set(color.keys()) == ckeys
            assert isinstance(color['name'], str)
            assert isinstance(color['bounds_ycrcb'], numpy.ndarray)
            assert color['bounds_ycrcb'].shape == (2, 3)
            assert color['bounds_ycrcb'].dtype == numpy.uint8
            self.color_names.append(color['name'])
            self.all_bboxes.append(color['bounds_ycrcb'])

        self.all_bboxes = numpy.array(self.all_bboxes, dtype=numpy.uint8)

        self.update_palette()

    def update_palette(self):

        display = (self.all_bboxes.astype(int).sum(axis=1) // 2).astype(numpy.uint8)
        display = cv2.cvtColor(display.reshape(-1, 1, 3), self.from_ycrcb).reshape(-1, 3)
        
        self.palette = numpy.vstack((display, 191*numpy.ones(3, dtype=numpy.uint8)))

        lpalette = numpy.zeros((256, 3), dtype=numpy.uint8)
        idx = (1 << numpy.arange(self.num_colors))
        lpalette[idx] = self.palette[:-1]
        lpalette[0] = self.palette[-1]

        self.full_palette = lpalette

    def save(self, config_filename=None):
        
        if config_filename is None:
            config_filename = self.json_filename

        colors = []

        for color_name, bbox in zip(self.color_names, self.all_bboxes):
            color = dict(name=color_name, bounds_ycrcb=bbox)
            colors.append(color)

        encoded = json_encode(json_from_numpy(dict(colors=colors)))

        with open(config_filename, 'w') as ostr:
            ostr.write(encoded)
            ostr.write('\n')

        print('wrote', config_filename)

    def convert_to_ycrcb(self, image):
        return cv2.cvtColor(image, self.to_ycrcb)

    def convert_from_ycrcb(self, image):
        return cv2.cvtColor(image, self.from_ycrcb)
    
    def label_image(self, image, dst=None):
        return label_image(self.all_bboxes, image, dst)

    def colorize_labels(self, labels):
        assert len(labels.shape) == 2 and labels.dtype == numpy.uint8
        return self.full_palette[labels]

    @property
    def num_colors(self):
        return len(self.color_names)

def _test_me():

    detector = ColorBlobDetector()

    for name, bbox in zip(detector.color_names, detector.all_bboxes):
        print('{} has bounds {}-{}'.format(name, bbox[0], bbox[1]))

    image = detector.palette.reshape(1, -1, 3)
    image_ycrcb = detector.convert_to_ycrcb(image)

    labels = detector.label_image(image_ycrcb).flatten()

    print('this should be in order:', labels[:-1])

    assert numpy.all(labels[:-1] == (1 << numpy.arange(detector.num_colors)))


if __name__ == '__main__':            
    _test_me()
    
