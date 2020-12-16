import numpy as np
import random
from scipy.io import loadmat

from ..logger import logger
from ..config import config, default
from ..dataset import *


def load_gt_roidb(dataset_name, image_set_name, dataset_path,
                  flip=False):
    """ load ground truth roidb """
    imdb = eval(dataset_name)(image_set_name, dataset_path)
    roidb = imdb.gt_roidb()
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def load_proposal_roidb(dataset_name, image_set_name, root_path, dataset_path,
                        proposal='rpn', append_gt=True, flip=False):
    """ load proposal roidb (append_gt when training) """
    imdb = eval(dataset_name)(image_set_name, root_path, dataset_path)
    gt_roidb = imdb.gt_roidb()
    roidb = eval('imdb.' + proposal + '_roidb')(gt_roidb, append_gt)
    if flip:
        roidb = imdb.append_flipped_images(roidb)
    return roidb


def merge_roidb(roidbs):
    """ roidb are list, concat them together """
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    return roidb


def filter_roidb(roidb):
    """ remove noisy bboxes, then remove roidb entries without usable rois """
    filtered_roidb = []
    num_boxes = 0
    num_boxes_new = 0
    for r in roidb:
        num_boxes += r['boxes'].shape[0]
        r['boxes'] = r['boxes'][r['noisy']==0]
        num_boxes_new += r['boxes'].shape[0]
        if r['boxes'].shape[0] > 0:
            filtered_roidb.append(r)
    logger.info('noisy boxes filtered, images: %d -> %d, bboxes: %d -> %d' %
                (len(roidb), len(filtered_roidb), num_boxes, num_boxes_new))

    return filtered_roidb

def sample_roidb(roidb, target):
    """ resamples the input set to a new random lower target (1-100)"""
    filtered_roidb = []
    num_boxes = 0
    num_boxes_new = 0
    for r in roidb:
        num_boxes += r['boxes'].shape[0]
        r['boxes'] = r['boxes'][r['noisy']==0]
        num_boxes_new += r['boxes'].shape[0]
        rval = random.randint(1,100)
        if rval <= target:
            filtered_roidb.append(r)
    logger.info('sample boxes filtered, images: %d -> %d, bboxes: %d -> %d' %
                (len(roidb), len(filtered_roidb), num_boxes, num_boxes_new))

    return filtered_roidb

def append_roidb(roidbsrc,roidbnew):
    """ concatenates two roidbs with selection criteria (target)"""
    appended_roidb = []
    num_boxes = 0
    num_boxes_new = 0

    imset = []
    # Select some of the src set to keep
    for r in roidbsrc:
        # For source, check if we're keeping this---
        # If r != positive
        # continue
        # else
        # Decide whether to promote this feature (rand annealing)
        rval = random.randint(1,100)
        if rval <= 5:
            continue
        
        # Add the current feature to our annealing list
        num_boxes += r['boxes'].shape[0]
        r['boxes'] = r['boxes'][r['noisy']==0]
        num_boxes_new += r['boxes'].shape[0]
        appended_roidb.append(r)

        # Build lookup table.
        #imset.append(r['image'])

    # Append all of new set
    for r in roidbnew:
        #print(r['image'])
        #print(appended_roidb)

        # only add unique sets
        found = False
        for s in appended_roidb:
            if r['image'] == s['image']:
                found = True
                break

        #print(found)
        if not found:
            num_boxes += r['boxes'].shape[0]
            r['boxes'] = r['boxes'][r['noisy']==0]
            num_boxes_new += r['boxes'].shape[0]
            appended_roidb.append(r)

        #Test code
        #key='image'
        #value = r['image']
        #key in a and value == a[key]
        #print( any(e['image'] == imname for e in appended_roidb) )
        #print( key in appended_roidb and value == appended_roidb[key] )
        #if r['image'] not in appended_roidb['image']: #maybe r['boxes']?
        #if key in appended_roidb and value == appended_roidb[key]:

    logger.info('appended boxes total, images: %d -> %d, bboxes: %d -> %d' %
                (len(roidbsrc), len(appended_roidb), num_boxes, num_boxes_new))

    return appended_roidb
