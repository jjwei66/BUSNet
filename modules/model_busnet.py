import torch
import torch.nn as nn
from fastai.vision import *

from .model_vision import BaseVision
from .model_language import BCNLanguage
from .model_semantic_visual_backbone_feature import BaseSemanticVisual_backbone_feature
from .model_vision_language import vision_language_reasoning_module, vision_language_reasoning_module_iter

class busnet_pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.test_bh = ifnone(config.test_bh, None)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision_language = vision_language_ver8_block_small_dec_base(config)

    # def forward(self, images, *args):
    def forward(self, images, texts=None):
        if self.training:
            assert isinstance(texts[-1], list), 'SpellMutation must used. Please check.'
            text_x, length_x = texts[-1][0],  texts[-1][1]
        else:
            text_x, length_x = None, None
        all_vl_res, all_l_res, all_v_res, all_a_res = [], [], [], []
        for _ in range(self.iter_size):
            l_res, v_res, vl_res = self.vision_language(images, text_x, length_x)  # l, v, vl
            all_l_res.append(l_res)
            all_v_res.append(v_res)
            all_vl_res.append(vl_res)


        if self.training:
            return all_vl_res, all_l_res, all_v_res
        else:
            return vl_res

class busnet_finetune(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.test_bh = ifnone(config.test_bh, None)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        # self.vision_language = vision_language_ver8_block_small_iter(config)
        self.vision_language = vision_language_ver8_block_iter(config)

    # def forward(self, images, *args):
    def forward(self, images, texts=None):
        all_vl_res, all_l_res, all_v_res, all_a_res = [], [], [], []
        for _ in range(self.iter_size):
            l_res, v_res, vl_res = self.vision_language(images)  # l, v, vl
            all_l_res.append(l_res)
            all_v_res.append(v_res)
            all_vl_res.append(vl_res)


        if self.training:
            return all_vl_res, all_l_res, all_v_res
        else:
            return vl_res


class busnet_finetune_iter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.test_bh = ifnone(config.test_bh, None)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision_language = vision_language_ver8_block_iter_training(config)

    # def forward(self, images, *args):
    def forward(self, images, texts=None):

        l_res, v_res, vl_res = self.vision_language(images)  # l, v, vl

        if self.training:
            return vl_res, l_res, v_res
        else:
            return vl_res[-1]

