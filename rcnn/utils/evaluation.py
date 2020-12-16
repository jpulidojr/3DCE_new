import numpy as np

#========================cwh======================
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
#========================cwh======================

from rcnn.config import config


def IOU(box1, gts):
	# compute overlaps
	# intersection
	ixmin = np.maximum(gts[:, 0], box1[0])
	iymin = np.maximum(gts[:, 1], box1[1])
	ixmax = np.minimum(gts[:, 2], box1[2])
	iymax = np.minimum(gts[:, 3], box1[3])
	iw = np.maximum(ixmax - ixmin + 1., 0.)
	ih = np.maximum(iymax - iymin + 1., 0.)
	inters = iw * ih

	# union
	uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
	       (gts[:, 2] - gts[:, 0] + 1.) *
	       (gts[:, 3] - gts[:, 1] + 1.) - inters)

	overlaps = inters / uni
	# ovmax = np.max(overlaps)
	# jmax = np.argmax(overlaps)
	return overlaps


def num_true_positive(boxes, gts, num_box, iou_th):
	# only count once if one gt is hit multiple times
	hit = np.zeros((gts.shape[0],), dtype=np.bool)
	scores = boxes[:, -1]
	boxes = boxes[scores.argsort()[::-1], :4]

	for i, box1 in enumerate(boxes):
		if i == num_box: break
		overlaps = IOU(box1, gts)
		hit = np.logical_or(hit, overlaps >= iou_th)

	tp = np.count_nonzero(hit)

	return tp


def recall_all(boxes_all, gts_all, num_box, iou_th):
	# Compute the recall at num_box candidates per image
	nCls = len(boxes_all)
	nImg = len(boxes_all[0])
	recs = np.zeros((nCls, len(num_box)))
	nGt = np.zeros((nCls,), dtype=np.float)

	for cls in range(nCls):
		for i in range(nImg):
			nGt[cls] += gts_all[cls][i].shape[0]
			for n in range(len(num_box)):
				tp = num_true_positive(boxes_all[cls][i], gts_all[cls][i], num_box[n], iou_th)
				recs[cls, n] += tp

	recs /= nGt
	return recs

def computeIDs(boxes_all, gts_all, iou_th):
	# Compute the ID's of box elements that are True Positive and Negative and return them
	nImg = len(boxes_all)
	img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)])
	boxes_cat = np.vstack(boxes_all)
	scores = boxes_cat[:, -1]
	ord = np.argsort(scores)[::-1]
	boxes_cat = boxes_cat[ord, :4]
	img_idxs = img_idxs[ord]
	hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]

	nHits = 0
	nMiss = 0

	tps = []
	fps = []

	for i in range(len(boxes_cat)):
		overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
		if overlaps.max() < iou_th:
			nMiss += 1
			fps.append(i)
		else:
			for j in range(len(overlaps)):
				if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
					hits[img_idxs[i]][j] = True
					nHits += 1
					tps.append(i)
                               

	print( "---False positive ids: ")
	print(fps)

	print( "---True positive ids: ")
	print(tps)	
		
	nGt = len(np.vstack(gts_all))
	print( "---Amount in GT: " , nGt )

    # Filter out boxes here and return array to feed into data loader
	

def FROC(boxes_all, gts_all, iou_th):
	# Compute the FROC curve, for single class only
	nImg = len(boxes_all)
	img_idxs = np.hstack([[i]*len(boxes_all[i]) for i in range(nImg)])
	boxes_cat = np.vstack(boxes_all)
	scores = boxes_cat[:, -1]
	ord = np.argsort(scores)[::-1]
	boxes_cat = boxes_cat[ord, :4]
	img_idxs = img_idxs[ord]
	hits = [np.zeros((len(gts),), dtype=bool) for gts in gts_all]

	nHits = 0
	nMiss = 0

	tps = []
	fps = []

	for i in range(len(boxes_cat)):
		overlaps = IOU(boxes_cat[i, :], gts_all[img_idxs[i]])
		if overlaps.max() < iou_th:
			nMiss += 1
		else:
			for j in range(len(overlaps)):
				if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
					hits[img_idxs[i]][j] = True
					nHits += 1
                               

		tps.append(nHits)
		fps.append(nMiss)


	nGt = len(np.vstack(gts_all))
	sens = np.array(tps, dtype=float) / nGt
	fp_per_img = np.array(fps, dtype=float) / nImg
        
#====================cwh=======================
	#add evaluation metric
	tps = np.array(tps, dtype=float)
	fps = np.array(fps, dtype=float)
	pre = tps/(tps + fps)
	pre = np.array(pre, dtype=float)  #precision
	rec = sens   #recall
	f = 2*pre*rec/(pre+rec)  #f-measure

	return sens, fp_per_img, pre, rec, f


from scipy import interpolate
def sens_at_FP(boxes_all, gts_all, avgFP, iou_th):
	# compute the sensitivity at avgFP (average FP per image)
	sens, fp_per_img, pre, rec, f = FROC(boxes_all, gts_all, iou_th)
	f1 = interpolate.interp1d(fp_per_img, sens, fill_value='extrapolate')
	

        # Similar to the interpolation of f1
        f2 = interpolate.interp1d(fp_per_img, pre, fill_value='extrapolate')
        f3 = interpolate.interp1d(fp_per_img, rec, fill_value='extrapolate')
        f4 = interpolate.interp1d(fp_per_img, f, fill_value='extrapolate')
        res1 = f1(np.array(avgFP))
        # Output the different metrics
        res2 = f2(np.array(avgFP))
        res3 = f3(np.array(avgFP))
        res4 = f4(np.array(avgFP))
	return res1, res2, res3, res4
#========================cwh======================
