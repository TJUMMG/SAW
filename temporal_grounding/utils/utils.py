import numpy as np
import torch.nn.functional as F
import torch


def gaussian_radius_1d(det_size, min_overlap=0.3):
    r1 = det_size * (1 - min_overlap) / 2
    r2 = det_size * (1 - min_overlap) / (2 * (1 + min_overlap))
    return min(r1, r2)


def gaussian1D(r, sigma=1):
    m = (r - 1.) / 2
    x = np.arange(-m, m+1)

    h = np.exp(-x * x / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_guassian(ct_int, total_length, radius):
    heatmap = np.zeros(total_length)
    diameter = 2 * radius + 1
    guassian = gaussian1D(diameter, diameter / 3)
    left, right = min(ct_int, radius), min(total_length - ct_int, radius + 1)
    heatmap[ct_int - left:ct_int + right] = guassian[radius-left:radius+right]
    return heatmap


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool1d(
        heat.unsqueeze(1), kernel, stride=1, padding=pad)
    hmax = hmax.squeeze(1)
    keep = (hmax == heat).float()
    return (heat * keep).squeeze(1)


def draw_dense_reg(total_length, heatmap, ct_int, value, radius):
    regmap = np.zeros(total_length)
    diameter = 2 * radius + 1
    gaussian = gaussian1D(diameter, diameter / 3)
    reg = np.ones(diameter*2+1, dtype=np.float32) * value
    left, right = min(ct_int, radius), min(total_length - ct_int, radius + 1)
    masked_heatmap = heatmap[ct_int - left:ct_int + right]
    masked_regmap = regmap[ct_int - left:ct_int + right]
    masked_gaussian = gaussian[radius - left:radius + right]
    masked_reg = reg[radius - left:radius + right]
    idx = (masked_gaussian >= masked_heatmap).reshape(masked_gaussian.shape[0])
    masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[ct_int - left:ct_int + right] = masked_regmap
    return regmap


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y


def iou(pred, gt):  # require pred and gt is numpy\
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    if not pred_is_list:
        pred = [pred]
    if not gt_is_list:
        gt = [gt]
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(0.0, union_right - union_left)
    overlap = 1.0 * inter / union
    if not gt_is_list:
        overlap = overlap[:, 0]
    if not pred_is_list:
        overlap = overlap[0]
    return overlap


def generate_2dtan(num_clips, duration, gt):
    gt_s_time, gt_e_time = gt[0], gt[1]
    s_times = torch.arange(0, num_clips).float() * duration / num_clips
    e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips
    overlaps = iou(torch.stack([s_times[:, None].expand(-1, num_clips),
                                e_times[None, :].expand(num_clips, -1)], dim=2).view(-1, 2).tolist(),
                   torch.tensor([gt_s_time, gt_e_time]).tolist()).reshape(num_clips, num_clips)
    return overlaps


def iou_tan(candidates, gt):
    start, end = candidates[:, 0], candidates[:, 1]
    s, e = gt[0].float(), gt[1].float()
    # print(s.dtype, start.dtype)
    inter = end.min(e) - start.max(s)
    union = end.max(e) - start.min(s)
    return inter.clamp(min=0) / union


def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero()
    scores = score2d[grids[:, 0], grids[:, 1]]
    grids[:, 1] += 1

    moments = grids * duration / num_clips
    return moments, scores


def nms(moments, scores, topk, thresh):
    scores, ranks = scores.sort(descending=True)

    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou_tan(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True

    return moments[~suppressed]


def generate_candidates(max_frames_num, window_widths):
    widths = np.array(window_widths)
    candidates = []
    for w in widths:
        for start in range(max_frames_num - w + 1):
            candidates.append([start, start + w - 1])
    return np.array(candidates), tuple(int(w) for w in widths)


def nms_detections(props, scores, overlap=0.7):

    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]

    out_proposals = []
    for idx in range(nms_props.shape[0]):
        prop = nms_props[idx][np.newaxis, :]
        score = float(nms_scores[idx])
        out_proposals.append(prop)

    return out_proposals

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


def get_iou(pred, gt):
    """ Get tIoU of two segments
    """
    start_pred, end_pred = pred
    start, end = gt
    intersection = max(0, min(end, end_pred) - max(start, start_pred))
    union = min(max(end, end_pred) - min(start, start_pred),
                end - start + end_pred - start_pred)
    iou = float(intersection) / (union + 1e-8)
    return iou


def get_recall_at_k(predictions, groundtruths, iou_threshold=0.5, max_proposal_num=5):
    """ Get R@k for all predictions
    R@k: Given k proposals, if there is at least one proposal has higher tIoU than iou_threshold, R@k=1; otherwise R@k=0
    The predictions should have been sorted by confidence
    """
    hit = np.zeros(shape=(len(groundtruths.keys()),), dtype=np.float32)

    for idd, idx in enumerate(groundtruths.keys()):
        if idx in predictions.keys():
            preds = predictions[idx][:max_proposal_num]
            for pred in preds:
                if calculate_IoU(pred['timestamp'], groundtruths[idx]['timestamp']) >= iou_threshold:
                    hit[idd] = 1.

    avg_recall = np.sum(hit) / len(hit)
    return avg_recall


def get_miou(predictions, groundtruths):
    """ Get mean IoU
    """
    ious = []
    for idx in groundtruths.keys():
        pred = predictions[idx][0]
        ious.append(calculate_IoU(
            pred['timestamp'], groundtruths[idx]['timestamp']))

    miou = sum(ious) / len(ious)

    return miou


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
