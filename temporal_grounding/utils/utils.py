import numpy as np
import torch


def generate_anchors(windows):
    widths = np.array(windows)
    center = 7.5
    start = center - 0.5 * (widths - 1)
    end = center + 0.5 * (widths - 1)
    return np.stack([start, end], -1)

def generate_proposals(max_num_frames, windows):
    anchors = generate_anchors(windows)
    widths = (anchors[:, 1] - anchors[:, 0] + 1)  # [num_anchors]
    centers = np.arange(0, max_num_frames)  # [video_len]
    start = np.expand_dims(centers, 1) - 0.5 * (np.expand_dims(widths, 0) - 1)
    end = np.expand_dims(centers, 1) + 0.5 * (np.expand_dims(widths, 0) - 1)
    proposals = np.stack([start, end], -1)  # [video_len, num_anchors, 2]
    return proposals

def generate_scores(proposals, label, max_num_frames, thres_score, thres_adjmat):
    proposals = np.reshape(proposals, [-1, 2])
    illegal = np.logical_or(proposals[:, 0] < 0, proposals[:, 1] >= max_num_frames)
    label1 = np.repeat(np.expand_dims(label, 0), proposals.shape[0], 0)
    IoUs = calculate_IoU_batch((proposals[:, 0], proposals[:, 1]),
                                        (label1[:, 0], label1[:, 1]))
    IoUs[illegal] = 0.0  # [video_len * num_anchors]
    max_IoU = np.max(IoUs)
    IoUs[IoUs < thres_score * max_IoU] = 0.0
    IoUs = IoUs / (max_IoU + 1e-4)
    adj_mat = IoUs.copy()
    adj_mat[adj_mat < thres_adjmat] = 0.0  # best 0.7 * max_IoU

    scores = IoUs.astype(np.float32)
    scores_mask = (1 - illegal).astype(np.uint8)
    return scores, scores_mask, adj_mat

def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou

def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))

    if union[1] - union[0] < -1e-5:
        return 0
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0] + 1e-10)
    return iou if iou >= 0.0 else 0.0


def average_to_fixed_length(visual_input, num_sample_clips):
    num_clips = visual_input.shape[0]
    idxs = torch.arange(0, num_sample_clips+1, 1.0)/num_sample_clips*num_clips
    idxs = torch.min(torch.round(idxs).long(), torch.tensor(num_clips-1))
    new_visual_input = []
    for i in range(num_sample_clips):
        s_idx, e_idx = idxs[i].item(), idxs[i+1].item()
        if s_idx < e_idx:
            new_visual_input.append(torch.mean(
                visual_input[s_idx:e_idx], dim=0))
        else:
            new_visual_input.append(visual_input[s_idx])
    new_visual_input = torch.stack(new_visual_input, dim=0)
    return new_visual_input


def nms_temporal(predict_score, predict_windows, overlap):
    pick = list()
    starts = predict_windows[:, 0]
    ends = predict_windows[:, 1]
    scores = predict_score
    assert len(starts) == len(scores)
    if len(starts) == 0:
        return pick

    unions = ends - starts
    # sort and get index
    indexs = [x[0] for x in sorted(enumerate(scores), key=lambda x:x[1])]

    while len(indexs) > 0:
        i = indexs[-1]
        pick.append(i)

        lefts = [max(starts[i], starts[j]) for j in indexs[:-1]]
        rights = [min(ends[i], ends[j]) for j in indexs[:-1]]
        inters = [max(0.0, right-left) for left, right in zip(lefts, rights)]
        laps = [inters[u]/(unions[i] + unions[indexs[u]] - inters[u])
                for u in range(len(indexs)-1)]
        indexs_new = []
        for j in range(len(laps)):
            if laps[j] <= overlap:
                indexs_new.append(indexs[j])
        indexs = indexs_new

    return pick


def compute_IoU_recall_top_n(predict_windows, gt_windows, picks, top_n, IoU_thresh):

    correct = 0
    if top_n < len(picks):
        cur_picks = picks[0:top_n]
    else:
        cur_picks = picks
    for index in cur_picks:
        pred_start = predict_windows[index][0]
        pred_end = predict_windows[index][1]
        iou = calculate_IoU(gt_windows, (pred_start, pred_end))
        if iou >= IoU_thresh:
            correct += 1
            break

    return correct


def compute_IoU_recall(predict_score, predict_windows, gt_windows):

    IoU_threshs = [0.1, 0.3, 0.5, 0.7]
    top_n_list = [1, 5]
    topn_IoU_matric = np.zeros([2, 4], dtype=np.float32)

    for i, IoU_thresh in enumerate(IoU_threshs):
        picks = nms_temporal(predict_score, predict_windows, IoU_thresh-0.05)

        for j, top_n in enumerate(top_n_list):
            correct = compute_IoU_recall_top_n(
                predict_windows, gt_windows, picks, top_n, IoU_thresh)
            topn_IoU_matric[j, i] = correct

    return topn_IoU_matric


class CountMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = np.zeros([2, 4], dtype=np.float32)
        self.count = 0

    def update(self, val, n=1):
        self.val += val
        self.count += n


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
