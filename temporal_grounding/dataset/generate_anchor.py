import numpy as np

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

def calculate_IoU_batch(i0, i1):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


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


def generate_candidates_scores(candidates, label, seq_len, duration):
    labels = np.repeat(np.expand_dims(label, 0), candidates.shape[0], 0)  # candidate_num x 2
    IoUs = calculate_IoU_batch(
        (candidates[:, 0] * duration / seq_len, (candidates[:, 1] + 1) * duration / seq_len),
        (labels[:, 0] * duration / seq_len, (labels[:, 1] + 1) * duration / seq_len)
    )
    max_IoU = np.max(IoUs)
    if max_IoU == 0.0:
        print(label)
        exit(1)
    IoUs[IoUs < 0.3 * max_IoU] = 0.0
    IoUs = IoUs / max_IoU
    scores = IoUs.astype(np.float32)
    return scores