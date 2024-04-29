import logging

import torch
import torch.nn as nn
from fastai.vision import *

from .model import _default_tfmer_cfg
from .model import Model
from .transformer import (PositionalEncoding,
                                 TransformerEncoder,
                                 TransformerDecoder,
                                 TransformerDecoderLayer, TransformerEncoderLayer,
                                _get_clones, _get_activation_fn)
from timm.models.vision_transformer import Block, PatchEmbed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class vision_language_reasoning_module(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_vision_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_vision_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_vision_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_vision_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_vision_language_activation, _default_tfmer_cfg['activation'])
        encoder_num_layers = ifnone(config.model_vision_language_encoder_num_layers, 12)
        decoder_num_layers = ifnone(config.model_vision_language_decoder_num_layers, 4)

        self.d_model = d_model
        self.detach = ifnone(config.model_vision_language_detach, True)
        self.use_self_attn = ifnone(config.model_vision_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_vision_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_proj = PatchEmbed(img_size=(32, 128), patch_size=(4, 8), in_chans=3, embed_dim=d_model)
        self.num_patches = self.patch_proj.num_patches
        self.patch_encoder = PositionalEncoding(d_model, max_len=self.num_patches)
        self.encoder = nn.ModuleList([
            Block(d_model, nhead, mlp_ratio=4., qkv_bias=True,
                  init_values=None, norm_layer=nn.LayerNorm)
            for _ in range(encoder_num_layers)])
        self.v_norm = nn.LayerNorm(d_model)

        # --------------------------------------------------------------------------
        # decoder specifics
        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout,
                activation, self_attn=self.use_self_attn, debug=self.debug)

        self.decoder = TransformerDecoder(decoder_layer, decoder_num_layers)

        v_mask = torch.empty((1, 1, d_model))
        l_mask = torch.empty((1, 1, d_model))
        self.v_mask = nn.Parameter(v_mask)
        self.l_mask = nn.Parameter(l_mask)
        torch.nn.init.uniform_(self.v_mask, -0.001, 0.001)
        torch.nn.init.uniform_(self.l_mask, -0.001, 0.001)

        v_embeding = torch.empty((1, 1, d_model))
        l_embeding = torch.empty((1, 1, d_model))
        self.v_embeding = nn.Parameter(v_embeding)
        self.l_embeding = nn.Parameter(l_embeding)
        torch.nn.init.uniform_(self.v_embeding, -0.001, 0.001)
        torch.nn.init.uniform_(self.l_embeding, -0.001, 0.001)

        self.cls = nn.Linear(d_model, self.charset.num_classes)

    def forward_decoder_language(self, q, x, padding_mask=None, location_mask=None):
        output = self.decoder(q, x,
                              tgt_key_padding_mask=padding_mask,
                              memory_mask=location_mask,
                              memory_key_padding_mask=padding_mask,
                              memory_key_padding_mask2=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
               'loss_weight': self.loss_weight, 'name': 'language'}
        return res

    def forward_decoder_vision(self, q, x, padding_mask=None, location_mask=None):
        output = self.decoder(q, x,
                              tgt_key_padding_mask=padding_mask,
                              memory_mask=location_mask,
                              memory_key_padding_mask=padding_mask,
                              memory_key_padding_mask2=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
               'loss_weight': self.loss_weight, 'name': 'vision_2'}
        return res

    def forward_decoder_vision_language(self, q, x, padding_mask=None, location_mask=None):
        output = self.decoder(q, x,
                              tgt_key_padding_mask=padding_mask,
                              memory_mask=location_mask,
                              memory_key_padding_mask=padding_mask,
                              memory_key_padding_mask2=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
               'loss_weight': self.loss_weight, 'name': 'vision_language'}
        return res

    def forward(self, images, tokens=None, lengths=None):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        # --------------------------------------------------------------------------
        # encoder procedure
        img_embed = self.patch_proj(images)
        N, L, E = img_embed.shape
        img_embed = img_embed.permute(1, 0, 2)  # L, N, E
        img_embed = self.patch_encoder(img_embed)
        img_embed = img_embed + self.v_embeding

        # img_feat = self.encoder(img_embed)

        img_embed = img_embed.permute(1, 0, 2)  # N, L, E
        img_feat = img_embed
        for blk in self.encoder:
            img_feat = blk(img_feat)
        img_feat = self.v_norm(img_feat)
        img_feat = img_feat.permute(1, 0, 2)  # L, N, E

        # --------------------------------------------------------------------------
        # decoder procedure
        T = self.max_length
        zeros = img_feat.new_zeros((T, N, E))
        zeros_len = img_feat.new_zeros(N)
        qeury = self.pos_encoder(zeros)

        location_mask = self._get_vl_location_mask(self.num_patches, self.max_length, img_feat.device)  # 对对应位置的

        # 1. vision decode
        v_embed = torch.cat((img_feat, self.l_mask.repeat(T, N, 1)), dim=0)  # v
        padding_mask = self._get_padding_mask(self.num_patches + zeros_len,
                                              self.num_patches + self.max_length)  # 对tokens长度以外的padding

        v_res = self.forward_decoder_vision(qeury, v_embed, padding_mask=padding_mask,
                                             location_mask=location_mask)

        # 2. language decode
        if tokens is None:
            tokens = torch.softmax(v_res['logits'], dim=-1)
            lengths = v_res['pt_lengths']
            tokens = tokens.detach()
        token_embed = self.proj(tokens)  # (N, T, E)
        token_embed = token_embed.permute(1, 0, 2)  # (T, N, E)
        token_embed = self.token_encoder(token_embed)  # (T, N, E)
        token_embed = token_embed + self.l_embeding

        padding_mask = self._get_padding_mask(self.num_patches + lengths,
                                              self.num_patches + self.max_length)  # 对tokens长度以外的padding

        l_embed = torch.cat((self.v_mask.repeat(L, N, 1), token_embed), dim=0)
        l_res = self.forward_decoder_language(qeury, l_embed, padding_mask=padding_mask,
                                             location_mask=location_mask)

        # 3. vision language decode
        vl_embed = torch.cat((img_feat, token_embed), dim=0)
        vl_res = self.forward_decoder_vision_language(qeury, vl_embed, padding_mask=padding_mask,
                                             location_mask=location_mask)

        return l_res, v_res, vl_res


class vision_language_reasoning_module_iter(Model):
    def __init__(self, config):
        super().__init__(config)
        d_model = ifnone(config.model_vision_language_d_model, _default_tfmer_cfg['d_model'])
        nhead = ifnone(config.model_vision_language_nhead, _default_tfmer_cfg['nhead'])
        d_inner = ifnone(config.model_vision_language_d_inner, _default_tfmer_cfg['d_inner'])
        dropout = ifnone(config.model_vision_language_dropout, _default_tfmer_cfg['dropout'])
        activation = ifnone(config.model_vision_language_activation, _default_tfmer_cfg['activation'])
        encoder_num_layers = ifnone(config.model_vision_language_encoder_num_layers, 12)
        decoder_num_layers = ifnone(config.model_vision_language_decoder_num_layers, 4)

        self.d_model = d_model
        self.detach = ifnone(config.model_vision_language_detach, True)
        self.use_self_attn = ifnone(config.model_vision_language_use_self_attn, False)
        self.loss_weight = ifnone(config.model_vision_language_loss_weight, 1.0)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.debug = ifnone(config.global_debug, False)

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_proj = PatchEmbed(img_size=(32, 128), patch_size=(4, 8), in_chans=3, embed_dim=d_model)
        self.num_patches = self.patch_proj.num_patches
        self.patch_encoder = PositionalEncoding(d_model, max_len=self.num_patches)
        self.encoder = nn.ModuleList([
            Block(d_model, nhead, mlp_ratio=4., qkv_bias=True,
                  init_values=None, norm_layer=nn.LayerNorm)
            for _ in range(encoder_num_layers)])
        self.v_norm = nn.LayerNorm(d_model)

        # --------------------------------------------------------------------------
        # decoder specifics
        self.proj = nn.Linear(self.charset.num_classes, d_model, False)
        self.token_encoder = PositionalEncoding(d_model, max_len=self.max_length)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0, max_len=self.max_length)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, d_inner, dropout,
                activation, self_attn=self.use_self_attn, debug=self.debug)

        self.decoder = TransformerDecoder(decoder_layer, decoder_num_layers)


        v_mask = torch.empty((1, 1, d_model))
        l_mask = torch.empty((1, 1, d_model))
        self.v_mask = nn.Parameter(v_mask)
        self.l_mask = nn.Parameter(l_mask)
        torch.nn.init.uniform_(self.v_mask, -0.001, 0.001)
        torch.nn.init.uniform_(self.l_mask, -0.001, 0.001)

        v_embeding = torch.empty((1, 1, d_model))
        l_embeding = torch.empty((1, 1, d_model))
        self.v_embeding = nn.Parameter(v_embeding)
        self.l_embeding = nn.Parameter(l_embeding)
        torch.nn.init.uniform_(self.v_embeding, -0.001, 0.001)
        torch.nn.init.uniform_(self.l_embeding, -0.001, 0.001)

        self.cls = nn.Linear(d_model, self.charset.num_classes)


    def forward_decoder_laguage(self, q, x, padding_mask=None, location_mask=None):
        output = self.decoder(q, x,
                              tgt_key_padding_mask=padding_mask,
                              memory_mask=location_mask,
                              memory_key_padding_mask=padding_mask,
                              memory_key_padding_mask2=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
               'loss_weight': self.loss_weight, 'name': 'language'}
        return res

    def forward_decoder_vision(self, q, x, padding_mask=None, location_mask=None):
        output = self.decoder(q, x,
                              tgt_key_padding_mask=padding_mask,
                              memory_mask=location_mask,
                              memory_key_padding_mask=padding_mask,
                              memory_key_padding_mask2=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
               'loss_weight': self.loss_weight, 'name': 'vision_2'}
        return res

    def forward_decoder_vision_laguage(self, q, x, padding_mask=None, location_mask=None):
        output = self.decoder(q, x,
                              tgt_key_padding_mask=padding_mask,
                              memory_mask=location_mask,
                              memory_key_padding_mask=padding_mask,
                              memory_key_padding_mask2=padding_mask)  # (T, N, E)
        output = output.permute(1, 0, 2)  # (N, T, E)

        logits = self.cls(output)  # (N, T, C)
        pt_lengths = self._get_length(logits)

        res = {'feature': output, 'logits': logits, 'pt_lengths': pt_lengths,
               'loss_weight': self.loss_weight, 'name': 'vision_language'}
        return res

    def language_tokens_encode(self, res, tokens=None, lengths=None):
        if tokens is None:
            tokens = torch.softmax(res['logits'], dim=-1)
            lengths = res['pt_lengths']
            tokens = tokens.detach()
        token_embed = self.proj(tokens)  # (N, T, E)
        token_embed = token_embed.permute(1, 0, 2)  # (T, N, E)
        token_embed = self.token_encoder(token_embed)  # (T, N, E)
        token_embed = token_embed + self.l_embeding

        return token_embed, lengths

    def forward(self, images, tokens=None, lengths=None):
        """
        Args:
            tokens: (N, T, C) where T is length, N is batch size and C is classes number
            lengths: (N,)
        """
        # --------------------------------------------------------------------------
        # encoder procedure
        img_embed = self.patch_proj(images)
        N, L, E = img_embed.shape
        img_embed = img_embed.permute(1, 0, 2)  # L, N, E
        img_embed = self.patch_encoder(img_embed)
        img_embed = img_embed + self.v_embeding

        img_embed = img_embed.permute(1, 0, 2)  # N, L, E
        img_feat = img_embed
        for blk in self.encoder:
            img_feat = blk(img_feat)
        img_feat = self.v_norm(img_feat)
        img_feat = img_feat.permute(1, 0, 2)  # L, N, E

        # --------------------------------------------------------------------------
        # decoder procedure
        T = self.max_length
        zeros = img_feat.new_zeros((T, N, E))
        zeros_len = img_feat.new_zeros(N)
        qeury = self.pos_encoder(zeros)

        location_mask = self._get_vl_location_mask(self.num_patches, self.max_length, img_feat.device)  # 对对应位置的

        # 1. vision decode
        v_embed = torch.cat((img_feat, self.l_mask.repeat(T, N, 1)), dim=0)  # v
        padding_mask = self._get_padding_mask(self.num_patches + zeros_len,
                                              self.num_patches + self.max_length)  # 对tokens长度以外的padding

        v_res = self.forward_decoder_vision(qeury, v_embed, padding_mask=padding_mask,
                                             location_mask=location_mask)

        # 2. language decode
        # token_embed, lengths = self.language_tokens_encode(v_res)
        # padding_mask = self._get_padding_mask(self.num_patches + lengths,
        #                                       self.num_patches + self.max_length)  # 对tokens长度以外的padding
        #
        # l_embed = torch.cat((self.v_mask.repeat(L, N, 1), token_embed), dim=0)
        # l_res = self.forward_decoder_laguage(qeury, l_embed, padding_mask=padding_mask,
        #                                      location_mask=location_mask)
        l_res = None

        # 3. vision language decode
        if self.training: iter = 1
        else: iter = 5
        vl_res = v_res
        for _ in range(iter):
            token_embed, lengths = self.language_tokens_encode(vl_res)
            padding_mask = self._get_padding_mask(self.num_patches + lengths,
                                                  self.num_patches + self.max_length)  # 对tokens长度以外的padding

            vl_embed = torch.cat((img_feat, token_embed), dim=0)
            vl_res = self.forward_decoder_vision_laguage(qeury, vl_embed, padding_mask=padding_mask,
                                                 location_mask=location_mask)

        return l_res, v_res, vl_res


