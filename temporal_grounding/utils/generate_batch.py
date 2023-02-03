import torch

def collect_fn(batch):
    """
    heat, offset, d, mask, embedding, fea, ratio, embedding_length, label
    """
    heats = []
    offsets = []
    ds = []
    ratios = []
    feas = []
    embeddings = []
    masks = []
    lengths = []
    labels = []
    max_fea_len = 0
    max_embedding_len = 0
    for b in batch:
        fea = b[5]
        embedding = b[4]
        lengths.append(b[7])
        if max_fea_len < fea.shape[-1]:
            max_fea_len = fea.shape[-1]
        if max_embedding_len < embedding.shape[0]:
            max_embedding_len = embedding.shape[0]
    lengths = torch.tensor(lengths)
    sorted, indices = lengths.sort(descending=True)
    for index in indices:
        b = batch[index]
        seq_length = b[0].shape[0]
        fea_padded = torch.zeros((b[5].shape[0], max_fea_len))
        embedding_padded = torch.zeros((max_embedding_len, b[4].shape[1]))
        heat_padded = torch.zeros(max_fea_len)
        mask_padded = torch.zeros(max_fea_len)
        d_padded = torch.zeros(max_fea_len)
        fea_padded[:, : seq_length] = b[5]
        embedding_padded[:b[4].shape[0], :] = b[4]
        mask_padded[: seq_length] = b[3]
        heat_padded[: seq_length] = b[0]
        d_padded[: seq_length] = b[2]

        offsets.append(b[1])
        ds.append(d_padded)
        feas.append(fea_padded)
        heats.append(heat_padded)
        embeddings.append(embedding_padded)
        masks.append(mask_padded)
        ratios.append(b[6])
        labels.append(b[-1])
    feas = torch.stack(feas, dim=0)
    heats = torch.stack(heats, dim=0)
    embeddings = torch.stack(embeddings, dim=0)
    masks = torch.stack(masks, dim=0)
    ratios = torch.stack(ratios, dim=0)
    offsets = torch.stack(offsets, dim=0)
    ds = torch.stack(ds, dim=0)
    labels = torch.stack(labels, dim=0)
    return heats, offsets, ds, masks, embeddings, feas, ratios, sorted, labels
