import torch
import torch.distributed as dist
import os
import numpy as np


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    dist_url = 'env://'
    print('| distributed init (rank {}): {}'.format(
        rank, dist_url), flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)



def calculate_IoU(pred, gt):
    SMOOTH = 1e-6
    IArea = (pred & (gt == 1.0)).astype(float).sum()
    OArea = (pred | (gt == 1.0)).astype(float).sum()
    IoU = (IArea + SMOOTH) / (OArea + SMOOTH)
    return IoU, IArea, OArea


def report_result(preds, labels):
    print(len(preds))
    MeanIoU, IArea, OArea, Overlap = [], [], [], []
    for i in range(len(preds)):
        iou, iarea, oarea = calculate_IoU(preds[i], labels[i])
        MeanIoU.append(iou)
        IArea.append(iarea)
        OArea.append(oarea)
        Overlap.append(iou)

    prec5, prec6, prec7, prec8, prec9 = np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1)), \
                                        np.zeros((len(Overlap), 1)), np.zeros((len(Overlap), 1))
    for i in range(len(Overlap)):
        if Overlap[i] >= 0.5:
            prec5[i] = 1
        if Overlap[i] >= 0.6:
            prec6[i] = 1
        if Overlap[i] >= 0.7:
            prec7[i] = 1
        if Overlap[i] >= 0.8:
            prec8[i] = 1
        if Overlap[i] >= 0.9:
            prec9[i] = 1

    mAP_thres_list = list(range(50, 95+1, 5))
    mAP = []
    for i in range(len(mAP_thres_list)):
        tmp = np.zeros((len(Overlap), 1))
        for j in range(len(Overlap)):
            if Overlap[j] >= mAP_thres_list[i] / 100.0:
                tmp[j] = 1
        mAP.append(tmp.sum() / tmp.shape[0])

    return np.mean(np.array(MeanIoU)), np.array(IArea).sum() / np.array(OArea).sum(), \
           prec5.sum() / prec5.shape[0], prec6.sum() / prec6.shape[0], prec7.sum() / prec7.shape[0], \
           prec8.sum() / prec8.shape[0], prec9.sum() / prec9.shape[0], np.mean(np.array(mAP))
