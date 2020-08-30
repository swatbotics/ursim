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

from .find_path import find_path

#SQRT22 = 0.5*numpy.sqrt(2)

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

def label_image(all_bboxes, image, labels=None, scratch=None):

    nlabels = len(all_bboxes)
    assert all_bboxes.shape == (nlabels, 2, 3)

    h, w = image.shape[:2]
    assert image.shape == (h, w, 3)

    assert nlabels <= 8

    init_mask = numpy.uint8((1 << nlabels) - 1)

    if labels is None:
        labels = numpy.empty((h, w), dtype=numpy.uint8)
    else:
        assert labels.shape == (h, w) and labels.dtype == numpy.uint8

    if (scratch is None or scratch.dtype != numpy.uint8 or scratch.shape != (h, w)):
        scratch = numpy.empty((h, w), dtype=numpy.uint8)
        
    labels[:] = init_mask

    for label_index in range(nlabels):
        label_mask = ~numpy.uint8(1 << label_index)
        for channel in range(3):
            ichan = image[:, :, channel]

            lo = all_bboxes[label_index, 0, channel]
            hi = all_bboxes[label_index, 1, channel]
            
            cv2.threshold(ichan, lo-1, label_mask, cv2.THRESH_BINARY_INV, scratch)
            cv2.bitwise_and(labels, scratch, labels, mask=scratch)

            cv2.threshold(ichan, hi, label_mask, cv2.THRESH_BINARY, scratch)
            cv2.bitwise_and(labels, scratch, labels, mask=scratch)
            
    return labels

######################################################################

class BlobDetection:

    def __init__(self, contour, area, xyz, is_split):

        self.contour = contour
        self.area = area
        self.xyz = xyz.copy()
        self.is_split = is_split

        decimated = self.xyz
        
        if len(decimated) > 500:
            decimated = decimated[::3]

        mean, principal_components, evals = cv2.PCACompute2(decimated, mean=None)

        evals = numpy.maximum(evals, 0)

        mean = mean.flatten()
        axes = 2*numpy.sqrt(evals.flatten())

        self.xyz_mean = mean
        self.axes = axes
        self.principal_components = principal_components

    def __str__(self):
        return('area fraction: {}'.format(self.area))

######################################################################

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
            self.json_filename = find_path('color_definitions.json')
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
    
    def label_image(self, image, dst=None, scratch=None):
        return label_image(self.all_bboxes, image, dst, scratch)

    def colorize_labels(self, labels):
        assert len(labels.shape) == 2 and labels.dtype == numpy.uint8
        return self.full_palette[labels]

    def detect_blobs(self,
                     labels,
                     min_contour_area=None,
                     xyz=None,
                     xyz_valid=None,
                     scratch=None,
                     split_axis=None,
                     split_res=None,
                     split_bins=None):

        if min_contour_area is None:
            min_contour_area = 0

        area_scl = 1.0 / numpy.prod(labels.shape)

        assert len(labels.shape) == 2 and labels.dtype == numpy.uint8
        assert xyz is None or xyz.shape == labels.shape + (3,)

        if (scratch is None or
            scratch.shape[0] < labels.shape[0] or
            scratch.shape[1] < labels.shape[1]):

            scratch = numpy.zeros_like(labels)

        detections = dict()

        for color_index in range(self.num_colors):

            color_name = self.color_names[color_index]

            mask = labels & (1 << color_index)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            color_detections = []

            for contour_index in range(len(contours)):

                contour = contours[contour_index]

                x0, y0, w, h = cv2.boundingRect(contour)
                
                if w*h*area_scl < min_contour_area:
                    continue

                topleft = (x0, y0)
                
                draw_mask = scratch[:h, :w]
                draw_mask[:] = 0


                shifted = contour - topleft

                cv2.drawContours(draw_mask, [shifted], 0,
                                 (255, 255, 255), -1)

                if xyz_valid is not None:
                    draw_mask = draw_mask & xyz_valid[y0:y0+h, x0:x0+w]
                
                area = numpy.count_nonzero(draw_mask)*area_scl
                if area < min_contour_area:
                    continue

                if xyz is None:
                    
                    detection = BlobDetection(contour, area,
                                              xyz=None, is_split=False)
                    
                    color_detections.append(detection)

                    continue
                
                xyz_subrect = xyz[y0:y0+h, x0:x0+w]

                mask_i, mask_j = numpy.nonzero(draw_mask)

                object_xyz = xyz_subrect[mask_i, mask_j]

                if (split_axis is None or split_res is None or split_bins is None):

                    detection = BlobDetection(contour, area,
                                              xyz=object_xyz, is_split = False)

                    color_detections.append(detection)

                    continue
                
                split_coords = object_xyz[:, split_axis]
                start_coord = split_coords.min()
                
                bin_idx = ((split_coords - start_coord) / split_res).astype(numpy.int32)

                unique_bins = numpy.unique(bin_idx)
                
                diffs = unique_bins[1:] - unique_bins[:-1]

                toobig = (diffs > split_bins)

                if not numpy.any(toobig):
                    
                    detection = BlobDetection(contour, area,
                                              object_xyz, is_split=False)
                    
                    color_detections.append(detection)

                else:

                    uidx0 = 0

                    while uidx0 < len(unique_bins):
                        
                        uidx1 = uidx0
                        
                        while uidx1 < len(toobig) and not toobig[uidx1]:
                            uidx1 += 1
                            
                        first_ok_bin = unique_bins[uidx0]
                        last_ok_bin = unique_bins[uidx1]

                        xidx, = numpy.nonzero((bin_idx >= first_ok_bin) &
                                              (bin_idx <= last_ok_bin))

                        area = len(xidx)*area_scl

                        if area >= min_contour_area:
                        
                            xi = mask_i[xidx]
                            xj = mask_j[xidx]

                            draw_mask[:] = 0
                            draw_mask[xi, xj] = 255

                            detection = BlobDetection(contour, area,
                                                      xyz_subrect[xi, xj],
                                                      is_split=True)

                            color_detections.append(detection)

                        uidx0 = uidx1 + 1

            color_detections.sort(key=lambda d: (-d.area, -d.contour[0,0,1]))

            detections[color_name] = color_detections

        return detections

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
    
