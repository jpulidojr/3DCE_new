import mxnet as mx
from rcnn.logger import logger
from rcnn.config import config, default
from rcnn.tools.test import test_rcnn
import numpy as np
import sys
import os
#===================>cwh new<===============================
from mxboard import SummaryWriter

# define a summary writer that logs data and flushes to the file every 5 seconds
#sw = SummaryWriter(logdir='./logs', flush_secs=5)

global_step = 0
#===================>cwh new<===============================

def validate(prefix, iter_no):    
    logger.info('Validating ...')
    default.testing = True
    ctx = mx.gpu(int(default.val_gpu))
    # ctx = mx.gpu(int(default.gpus.split(',')[0]))
    epoch = iter_no + 1
    # ===>cwh new<===
    acc,pre,rec,f = test_rcnn(default.network, default.dataset, default.val_image_set,
                      default.dataset_path,
                      ctx, prefix, epoch,
                      default.val_vis, default.val_shuffle,
                      default.val_has_rpn, default.proposal,
                      default.val_max_box, default.val_thresh)
    # ===>cwh new<===
    fn = '%s-%04d.params' % (prefix, epoch)
    fn_to_del = None
    if len(default.accs.keys()) == 0:
        default.best_model = fn
        default.best_acc = acc
        default.best_epoch = epoch
    else:
        if acc > default.best_acc:
            fn_to_del = default.best_model
            default.best_model = fn
            default.best_acc = acc
            default.best_epoch = epoch
        else:
            fn_to_del = fn

    default.accs[str(epoch)] = acc
    epochs = np.sort([int(a) for a in default.accs.keys()]).tolist()
    acc=0
   
    for e in epochs:
        print 'Iter %s: %.4f' % (e, default.accs[str(e)])
        # logging training accuracy       
        acc=default.accs[str(e)]
    sys.stdout.flush()

    if default.keep_best_model and fn_to_del:
        os.remove(fn_to_del)
        print fn_to_del, 'deleted to keep only the best model'
        sys.stdout.flush()

    default.testing = False
    return acc


#===================>cwh new<===============================
'''
    sw.add_scalar(tag='precision', value=('fp-0.5',pre[0]), global_step=iter_no)
    sw.add_scalar(tag='precision', value=('fp-1',pre[1]), global_step=iter_no)
    sw.add_scalar(tag='precision', value=('fp-2',pre[2]), global_step=iter_no)
    sw.add_scalar(tag='precision', value=('fp-4',pre[3]), global_step=iter_no)
    sw.add_scalar(tag='precision', value=('fp-8',pre[4]), global_step=iter_no)
    sw.add_scalar(tag='precision', value=('fp-16',pre[5]), global_step=iter_no)

    sw.add_scalar(tag='recall', value=('fp-0.5',rec[0]), global_step=iter_no)
    sw.add_scalar(tag='recall', value=('fp-1',rec[1]), global_step=iter_no)
    sw.add_scalar(tag='recall', value=('fp-2',rec[2]), global_step=iter_no)
    sw.add_scalar(tag='recall', value=('fp-4',rec[3]), global_step=iter_no)
    sw.add_scalar(tag='recall', value=('fp-8',rec[4]), global_step=iter_no)
    sw.add_scalar(tag='recall', value=('fp-16',rec[5]), global_step=iter_no)

    sw.add_scalar(tag='f-measure', value=('fp-0.5',f[0]), global_step=iter_no)
    sw.add_scalar(tag='f-measure', value=('fp-1',f[1]), global_step=iter_no)
    sw.add_scalar(tag='f-measure', value=('fp-2',f[2]), global_step=iter_no)
    sw.add_scalar(tag='f-measure', value=('fp-4',f[3]), global_step=iter_no)
    sw.add_scalar(tag='f-measure', value=('fp-8',f[4]), global_step=iter_no)
    sw.add_scalar(tag='f-measure', value=('fp-16',f[5]), global_step=iter_no)
'''
#===================>cwh new<===============================


