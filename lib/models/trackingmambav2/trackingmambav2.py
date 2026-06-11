
import math
import os
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.trackingmambav2.hivit import hivit_small, hivit_base
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.models.layers.transformer_dec import build_transformer_dec
from lib.models.layers.position_encoding import build_position_encoding
from lib.utils.misc import NestedTensor
#from lib.models.trackingmambav2.models_roma import create_block
#from lib.models.trackingmambav2.models_simba import SiMBA
from lib.models.trackingmambav2.models_mamba import create_block
from timm.models import create_model
from lib.config.trackingmambav2.config import cfg


def _resolve_backbone_pretrained(cfg):
    candidates = []
    env_path = os.environ.get("TRACKINGMAMBAV2_BACKBONE_PRETRAIN")
    if env_path:
        candidates.append(env_path)
    candidates.extend([
        "/home/zly/projects/pythonprojects/pretrained_models/hustvlVim-small-midclstok+/vim_s_midclstok_ft_81p6acc.pth",
        "/home/zly/projects/pythonprojects/pretrained_models/hustvlVim-small-midclstok+/vim_s_midclstok_80p5acc.pth",
        str(Path(__file__).resolve().parents[3] / "pretrained_models" /
            "hustvlVim-small-midclstok+" / "vim_s_midclstok_ft_81p6acc.pth"),
        str(Path(__file__).resolve().parents[3] / "pretrained_models" /
            "hustvlVim-small-midclstok+" / "vim_s_midclstok_80p5acc.pth"),
    ])
    pretrain_file = getattr(cfg.MODEL, "PRETRAIN_FILE", "")
    if pretrain_file and "TrackingmambaV2" not in pretrain_file:
        root = Path(__file__).resolve().parents[3] / "pretrained_models"
        candidates.append(str(root / pretrain_file))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""


class TrackingmambaV2(nn.Module):
    """ This is the base class for TrackingmambaV2 """

    def __init__(self, transformer, box_head, transformer_dec, position_encoding, aux_loss=False, head_type="CORNER"):

        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.transformer_dec = transformer_dec
        self.position_encoding = position_encoding
        hidden_dim = getattr(transformer_dec, "d_model", 384)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.memory_scale = nn.Parameter(torch.tensor(0.5))
        self.history_update_low = 0.20
        self.history_update_high = 0.55


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                return_last_attn=False,
                training=True, #True
                tgt_pre=None,
                ):
        template_list = [template] if torch.is_tensor(template) else template
        search_list = [search] if torch.is_tensor(search) else search
        b0, num_search = template_list[0].shape[0], len(search_list)

        if training:
            search = torch.cat(search_list, dim=0)
            template = template_list[0].repeat(num_search, 1, 1, 1)
        else:
            search = search_list[0]
            template = template_list[0]

        #print(tgt_pre.size())

        # x, aux_dict = self.backbone.forward_features(z=template, x=search,
        #                             inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False ) #x=[B,N,C]
        x, aux_dict = self.backbone.forward_features(z=template, x=search,
                                                     inference_params=None, if_random_cls_token_position=False,
                                                     if_random_token_rank=False)
        
        input_dec = x
        x_decs = []
        history_out = tgt_pre

        if training:
            batches = [[] for _ in range(b0)]
            for i, input_feature in enumerate(input_dec):
                batches[i % b0].append(input_feature.unsqueeze(0))

            per_sequence_decs = []
            for batch in batches:
                historical = None
                sequence_decs = []
                for input_feature in batch:
                    decoded, historical = self.transformer_dec(
                        input_feature.transpose(0, 1), historical)
                    sequence_decs.append(decoded)
                per_sequence_decs.append(sequence_decs)
            x_decs = [
                per_sequence_decs[j][i]
                for i in range(num_search)
                for j in range(b0)
            ]
            x_dec = torch.cat(x_decs, dim=1)
        else:
            if isinstance(history_out, (list, tuple)):
                history_out = None
            decoded, proposed_history = self.transformer_dec(
                input_dec.transpose(0, 1), history_out)
            x_dec = decoded

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        out = self.forward_head(feat_last, x_dec, None) # STM and head
        if not training:
            history_out = self.update_history_with_confidence(
                history_out, proposed_history, out.get('score_map'))

        out.update(aux_dict)
        out['tgt'] = history_out
        return out

    def update_history_with_confidence(self, history, proposed_history, score_map):
        proposed_history = proposed_history.detach()
        if history is None or isinstance(history, (list, tuple)) or history.shape != proposed_history.shape:
            return proposed_history
        if score_map is None:
            return proposed_history

        score_tokens = score_map.detach().flatten(2).transpose(1, 2)  # B, Ns, 1
        confidence = score_tokens.max(dim=1, keepdim=True)[0]
        update_ratio = ((confidence - self.history_update_low) /
                        (self.history_update_high - self.history_update_low)).clamp(0.0, 1.0)

        denom = confidence.clamp_min(1e-6)
        spatial_weight = (score_tokens / denom).clamp(0.0, 1.0)
        spatial_weight = 0.2 + 0.8 * spatial_weight

        num_search_tokens = self.feat_len_s
        num_template_tokens = proposed_history.shape[0] - num_search_tokens
        template_weight = update_ratio.expand(-1, num_template_tokens, -1)
        search_weight = update_ratio * spatial_weight
        token_weight = torch.cat([template_weight, search_weight], dim=1).permute(1, 0, 2)
        token_weight = token_weight.to(device=proposed_history.device, dtype=proposed_history.dtype)
        return history.detach() + token_weight * (proposed_history - history.detach())

    def forward_head(self, cat_feature, out_dec=None, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # STM
        enc_opt = cat_feature[:, -self.feat_len_s:]
        if out_dec is None:
            opt_feat = enc_opt.transpose(1, 2).contiguous().view(
                enc_opt.shape[0], enc_opt.shape[2], self.feat_sz_s, self.feat_sz_s)
            bs, Nq = enc_opt.shape[0], 1
        else:
            dec_tokens = out_dec.transpose(0, 1)
            dec_search = dec_tokens[:, -self.feat_len_s:]
            att = torch.matmul(enc_opt, dec_search.transpose(1, 2)) / math.sqrt(enc_opt.shape[-1])
            att = att.softmax(dim=-1)
            opt = torch.matmul(att, dec_search)
            opt = self.memory_norm(enc_opt + torch.sigmoid(self.memory_scale) * opt)
            bs, _, C = opt.size()
            Nq = 1
            opt_feat = opt.transpose(1, 2).contiguous().view(bs, C, self.feat_sz_s, self.feat_sz_s)

        #Head
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_trackingmambav2(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    # if cfg.MODEL.PRETRAIN_FILE and ('TrackingmambaV2' not in cfg.MODEL.PRETRAIN_FILE) and training:
    #     pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    # else:


    # if cfg.MODEL.BACKBONE.TYPE == 'hivit_small':
    #     backbone = hivit_small(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    # elif cfg.MODEL.BACKBONE.TYPE == 'hivit_base':
    #     backbone = hivit_base(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    # else:
    #     raise NotImplementedError
    #pretrained = '/home/zly/projects/pythonprojects/TrackingMamba_ROMA/mamba-base-400w-400epoch.pth'
    #pretrained = '/home/zly/projects/pythonprojects/TrackingMambaV2_2/pre/checkpoint-195.pth.tar'
    #pretrained = '/home/zhouliyao/projects/pythonprojects/OSMTrack2-main/pretrained_models/hustvlVim-small-midclstok+/vim_s_midclstok_80p5acc.pth'
    #pretrained = '/home/zly/projects/pythonprojects/pretrained_models/hustvlVim-small-midclstok+/vim_s_midclstok_80p5acc.pth'

    # backbone = create_model(model_name="simba_b", pretrained=pretrained, num_classes=1000,
    #                         drop_rate=0.0, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, drop_block_rate=None, img_size=256
    #                         )

    pretrained = _resolve_backbone_pretrained(cfg)
    backbone = create_model(model_name="vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2", pretrained=pretrained)

    #backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    hidden_dim = 384
    transformer_dec = build_transformer_dec(cfg, hidden_dim)
    position_encoding = build_position_encoding(cfg, sz = 1)

    box_head = build_box_head(cfg, hidden_dim)
    model = TrackingmambaV2(
        backbone,
        box_head,
        transformer_dec,
        position_encoding,        
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'TrackingmambaV2' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

if __name__ == '__main__':
    net = build_trackingmambav2(cfg)
    net = net.cuda()
    var1 = torch.Tensor(1, 3, 128, 128).cuda()
    var2 = torch.Tensor(1, 3, 256, 256).cuda()

    out = net(var1, var2)
    print("over")
