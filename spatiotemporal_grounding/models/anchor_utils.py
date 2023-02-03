import torch

def generate_anchors(windows):
    widths = torch.tensor(windows)
    center = 7.5
    start = center - 0.5 * (widths - 1)
    end = center + 0.5 * (widths - 1)
    return torch.stack([start, end], -1)

def generate_proposals(max_num_frames, windows):
    anchors = generate_anchors(windows)
    widths = (anchors[:, 1] - anchors[:, 0] + 1)  # [num_anchors]
    centers = torch.arange(0, max_num_frames)  # [video_len]
    start = centers[:, None] - 0.5 * (widths[None, :] - 1)
    end = centers[:, None] + 0.5 * (widths[None, :] - 1)
    proposals = torch.stack([start, end], -1)  # [video_len, num_anchors, 2]
    return proposals.view(-1, 2)

def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0]) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def generate_scores(proposals, label, max_num_frames, thres_score):
    illegal = torch.logical_or(proposals[:, 0] < 0, proposals[:, 1] >= max_num_frames)
    label1 = label[None, :].repeat(proposals.shape[0], 1)
    # label1 = np.repeat(np.expand_dims(label, 0), proposals.shape[0], 0)
    IoUs = calculate_IoU_batch((proposals[:, 0], proposals[:, 1]),
                                        (label1[:, 0], label1[:, 1]))
    IoUs[illegal] = 0.0  # [video_len * num_anchors]
    max_IoU = torch.max(IoUs)
    IoUs[IoUs < thres_score * max_IoU] = 0.0
    IoUs = IoUs / (max_IoU + 1e-4)

    scores = IoUs.float()
    scores_mask = (1 - illegal.float())
    return scores, scores_mask

def generate_2d_gaussian(boxes, w, h):
    # boxes: k*4    cxcywh
    n_boxes = len(boxes)
    ww = torch.linspace(0, 1, w)
    hh = torch.linspace(0, 1, h)
    gridh, gridw = torch.meshgrid(hh, ww)
    grid = torch.stack([gridw, gridh], dim=0)[None, ...].repeat(n_boxes, 1, 1, 1).to(boxes.device)  # k*2*h*w
    boxes = boxes[..., None, None].repeat(1, 1, h, w)
    gaussian = torch.exp(-(boxes[:, 0]-grid[:, 0])**2/(0.05*boxes[:, 2]**2))*\
                torch.exp(-(boxes[:, 1]-grid[:, 1])**2/(0.05*boxes[:, 3]**2))  # k*h*w
    gaussian[gaussian<0.05] = 0
    return gaussian
