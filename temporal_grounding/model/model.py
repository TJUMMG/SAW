import torch
import torch.nn as nn
import torch.nn.functional as F
from .module.attention import GlobalTextPresentation
from .module.RefTransformer import RefTransformer
from .decoder import AnchorBasedDecoder, RegressionDecoder


class TextEncoder(nn.Module):
    def __init__(self, config):
        self.config = config
        super(TextEncoder, self).__init__()
        if config['gru_bidirection']:
            self.backbone = nn.GRU(
                config['embedding_dim'], config['attention_dim'], batch_first=True, bidirectional=True)
        else:
            self.backbone = nn.GRU(
                config['embedding_dim'], config['attention_dim'], batch_first=True, bidirectional=False)
        self.dropout0 = nn.Dropout(config['dropout'])
        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, text, embedding_length):
        text = torch.nn.utils.rnn.pack_padded_sequence(
            text, list(embedding_length), True, enforce_sorted=False)
        word_embedding, _ = self.backbone(text)
        word_embedding, _ = torch.nn.utils.rnn.pad_packed_sequence(
            word_embedding, True)
        if self.config['gru_bidirection']:
            word_embedding = word_embedding.view(
                word_embedding.shape[0], word_embedding.shape[1], 2, -1)
            word_embedding = torch.mean(word_embedding, dim=2)
        return word_embedding


class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.video_encoder = nn.Linear(
            config['video_fea_dim'], config['attention_dim'])
        self.global_text = GlobalTextPresentation(config['attention_dim'])
        self.pos_embedding = nn.Parameter(torch.randn(
            1, config['attention_dim'], config['segment_num']))
        self.prenorm = nn.LayerNorm(config['attention_dim'])

        self.TCN = RefTransformer(config)
        if config['decoder_type'] == 'anchor_based':
            self.decoder = AnchorBasedDecoder(config)
        elif config['decoder_type'] == 'regression':
            self.decoder = RegressionDecoder(config)

    def forward(self, video_fea, embedding, embedding_length, score=None, gt_reg=None, score_mask=None, score_nm=None, proposals=None, adj_mat=None, mode='train'):
        text_feal = self.text_encoder(
            embedding.float(), embedding_length)  # b*l*c
        embedding_mask = torch.zeros((text_feal.shape[0], 1, text_feal.shape[1])).to(
            text_feal.device)

        for b in range(embedding_mask.shape[0]):
            embedding_mask[b, :, :int(embedding_length[b])] = 1
        text_feag, text_weight = self.global_text(
            text_feal, embedding_mask)  # b*1*d

        video_fea = self.video_encoder(video_fea.float())  # b*c*t
        if self.config['with_text']:
            video_fea = video_fea + text_feag

        video_fea = video_fea.permute(0, 2, 1)
        out_fea, weights = self.TCN(
            video_fea, text_feag, self.pos_embedding, embedding_mask)  # b*c*t
        if self.config['decoder_type'] == 'anchor_based':
            return self.decoder(out_fea, weights, score, gt_reg, score_mask, score_nm, proposals, adj_mat, mode)
        elif self.config['decoder_type'] == 'regression':
            return self.decoder(out_fea, weights, gt_reg, score_nm, mode)


if __name__ == '__main__':
    import json
    feav = torch.randn((4, 1024, 100))
    feat = torch.randn((4, 20, 300))
    with open('../json/config.json') as f:
        config = json.load(f)['model_config']
    model = Model(config)
    fea, weight = model(feav, feat, torch.tensor([15, 13, 12, 10]))
    print(fea.shape)
    print(len(weight))
