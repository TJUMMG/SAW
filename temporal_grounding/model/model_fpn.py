import torch
import torch.nn as nn
import torch.nn.functional as F
from .module.attention import GlobalTextPresentation
from .module.TCN import TCN, TCN2D
from .module.tanmodule import SparseMaxPool, generate_mask
from .decoder import AnchorBasedDecoderFPN



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


class VideoEncoder(nn.Module):
    def __init__(self, config):
        self.config = config
        super(VideoEncoder, self).__init__()
        if config['gru_bidirection']:
            self.backbone = nn.GRU(
                config['video_fea_dim'], config['attention_dim'], batch_first=True, bidirectional=True)
        else:
            self.backbone = nn.GRU(
                config['video_fea_dim'], config['attention_dim'], batch_first=True, bidirectional=False)

    def forward(self, fea_video):
        fea_video, _ = self.backbone(fea_video)
        if self.config['gru_bidirection']:
            fea_video = fea_video.view(
                fea_video.shape[0], fea_video.shape[1], 2, -1)
            fea_video = torch.mean(fea_video, dim=2)
        return fea_video


class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.video_encoder = nn.Conv1d(
            config['video_fea_dim'], config['attention_dim'], 1, 1)

        self.scale0 = nn.Sequential(
            nn.Conv1d(config['attention_dim'], config['attention_dim'], 1, 1, bias=False),
            nn.GroupNorm(4, config['attention_dim']),
            nn.ReLU()
        )
        self.scale1 = nn.Sequential(
            nn.Conv1d(config['attention_dim'], config['attention_dim'], 3, 2, 1, bias=False),
            nn.GroupNorm(4, config['attention_dim']),
            nn.ReLU(),
            nn.Conv1d(config['attention_dim'], config['attention_dim'], 1, 1),
            nn.GroupNorm(4, config['attention_dim']),
            nn.ReLU()
        )
        self.scale2 = nn.Sequential(
            nn.Conv1d(config['attention_dim'], config['attention_dim'], 3, 2, 1, bias=False),
            nn.GroupNorm(4, config['attention_dim']),
            nn.ReLU(),
            nn.Conv1d(config['attention_dim'], config['attention_dim'], 1, 1),
            nn.GroupNorm(4, config['attention_dim']),
            nn.ReLU()
        )

        self.global_text = GlobalTextPresentation(config['attention_dim'])
        self.pos_embedding = nn.Parameter(torch.randn(
            1, config['attention_dim'], config['segment_num']))
        self.TCN0 = TCN(config)
        self.TCN1 = TCN(config)
        self.TCN2 = TCN(config)
        self.decoder = AnchorBasedDecoderFPN(config)


    def forward(self, video_fea, embedding, embedding_length, gt_hm=None, gt_dis=None, score=None, gt_reg=None, score_mask=None, score_nm=None, proposals=None, tan_map=None, duration=None, mode='train'):

        text_feal = self.text_encoder(embedding.float(), embedding_length)  # b*l*c
        embedding_mask = torch.zeros((text_feal.shape[0], 1, text_feal.shape[1])).to(
            text_feal.device)
        for b in range(embedding_mask.shape[0]):
            embedding_mask[b, :, :int(embedding_length[b])] = 1
        text_feag, text_weight = self.global_text(
            text_feal, embedding_mask)  # b*1*d
        video_fea = self.video_encoder(video_fea.float().permute(
            0, 2, 1))  # b*c*t

        video_fea = video_fea + text_feag.permute(0, 2, 1) + self.pos_embedding

        fea0 = self.scale0(video_fea)
        fea1 = self.scale1(fea0)
        fea2 = self.scale2(fea1)

        out_fea0, weights0 = self.TCN(
            fea0, text_feag, text_feal, embedding_mask)  # b*c*t
        out_fea1, weights1 = self.TCN(
            fea1, text_feag, text_feal, embedding_mask)  # b*c*t
        out_fea2, weights2 = self.TCN(
            fea2, text_feag, text_feal, embedding_mask)  # b*c*t
        feas = [out_fea0, out_fea1, out_fea2]
        weights = [weights0, weights1, weights2]

        return self.decoder(feas, weights, score, gt_reg, score_mask, score_nm, proposals, mode)



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
