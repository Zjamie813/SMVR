from .clip_model import Transformer, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import copy
import torch.nn.functional as F
from einops import rearrange, repeat

# -----自定义包引入-------#
from model import objectives
from tools.model.Perceiver.perceiver_aggregation import PerceiverResampler
from tools.objectives.mvl_similarity import SetwiseSimilarity
from .compute_soft_labels import get_y_value_with_inter_cr, get_final_y_label_with_inter_intra_cr


class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'm2m' in args.loss_names:
            #------multi-view feature extraction -------#
            self.sim_type = args.sim_type
            self.global_fuse_mode = args.global_fuse_mode
            self.many_loss_weight = args.many_loss_weight
            ## --- visual features------#
            self.perceiver_softmax_mode = args.perceiver_softmax_mode
            self.set_prediction_module_image = PerceiverResampler(dim=self.embed_dim,
                                                                  depth=args.perceiver_image_depth,
                                                                  heads=args.perceiver_image_head,
                                                                  num_latents=args.perceiver_image_num_set,
                                                                  ff_mult=args.perceiver_image_ff_mult,
                                                                  dim_head=args.perceiver_image_dim_head,
                                                                  variance_constant=args.variance_constant,
                                                                  )
            # 设置模型的参数类型都为float32，不然由于本实验特殊设置会有些变为half
            self.set_prediction_module_image.type(torch.float16)
            self.norm_res_local_image = nn.LayerNorm(self.embed_dim)


            #--------SC similarity function--------#
            self.many2many_logit_scale = torch.ones([]) * (1 / args.mvl_loss_temperature)
            self.img_view_num = args.perceiver_image_num_set
            self.txt_view_num = 1
            self.sc_sim_temp = args.sc_sim_temp
            self.many_sim_func = SetwiseSimilarity(img_set_size=self.img_view_num, txt_set_size=self.txt_view_num,
                                                   denominator=2, temperature=self.sc_sim_temp)
            self.visual_sim_func = SetwiseSimilarity(img_set_size=self.img_view_num, txt_set_size=self.img_view_num,
                                                   denominator=2, temperature=self.sc_sim_temp)
            self.txt_sim_func = SetwiseSimilarity(img_set_size=self.txt_view_num, txt_set_size=self.txt_view_num,
                                                   denominator=2, temperature=self.sc_sim_temp)

            # inter label and intra label settings ------------#
            self.cr_beta = args.cr_beta
            self.ini_label_epoch = args.ini_label_epoch
            self.weight_sum_temperature = args.weight_sum_temperature


        if 'div' in args.loss_names:
            self.div_loss_weight = args.div_loss_weight

        if 'mlm' in args.loss_names:
            self.kv_img = args.kv_img
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)  # 原始的特征
        x_global = x[:, 0, :]
        x_local = x[:, 1:, :]
        image_set_embeddings = self.set_prediction_module_image(x_local, softmax_mode=self.perceiver_softmax_mode) # [bt, 4, dim]
        if self.global_fuse_mode == "without_global":
            image_res_set_embeddings = self.norm_res_local_image(image_set_embeddings.float())
        elif self.global_fuse_mode == "default":
            image_res_set_embeddings = self.norm_res_local_image(image_set_embeddings.float() + x_global.unsqueeze(1))
        return image_res_set_embeddings, x
        # return x.float() # for CLIP ResNet visual model

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        x_global = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        txt_set_embs = x_global.unsqueeze(1).float()
        return txt_set_embs

    def compute_set_similarity(self, image_list, text_list, sim_func):
        # image_list: [bt, len, dim], text_list:[bt, len, dim]
        batch_size, img_feat_len, dim = image_list.shape
        image_list = image_list.reshape(-1, dim)
        text_list = text_list.reshape(-1, dim)  # [bt*len, dim]

        image_list_norm = image_list / image_list.norm(dim=1, keepdim=True)
        text_list_norm = text_list / text_list.norm(dim=1, keepdim=True)

        i2t_cosine_theta = sim_func.smooth_chamfer_similarity_cosine(image_list_norm, text_list_norm)

        return i2t_cosine_theta

    def compute_LSE_board(self):
        min_sim = torch.zeros(3, 1)
        max_sim = torch.ones(3,1)
        min_sim_tao = min_sim * self.sc_sim_temp
        max_sim_tao = max_sim * self.sc_sim_temp
        min = (torch.sum(torch.logsumexp(min_sim_tao, dim=1)) / (self.img_view_num * self.sc_sim_temp) +
               torch.sum(torch.logsumexp(min_sim_tao, dim=0)) / (self.txt_view_num * self.sc_sim_temp)) / 2
        max = (torch.sum(torch.logsumexp(max_sim_tao, dim=1)) / (self.img_view_num * self.sc_sim_temp) +
               torch.sum(torch.logsumexp(max_sim_tao, dim=0)) / (self.txt_view_num * self.sc_sim_temp)) / 2
        return min, max

    def forward(self, batch, epoch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        i_set_feats, mlm_img_feats = self.encode_image(images)  # [bt, 4, dim]
        t_set_feats = self.encode_text(caption_ids) # [bt, 1, dim]
        ret.update({'temperature': 1 / self.many2many_logit_scale})

        
        if 'id' in self.current_task:
            #------expand the image and text features as the same dimension [bt*view_num, dim]
            i_feats = i_set_feats.reshape(-1, self.embed_dim) # [bt*view_num, dim]
            t_feats = t_set_feats.repeat(1, self.img_view_num, 1).reshape(-1, self.embed_dim)
            labels = batch['pids'].unsqueeze(1).repeat(1, self.img_view_num).reshape(-1)
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            id_loss = objectives.compute_id(image_logits, text_logits, labels)
            ret.update({'id_loss':id_loss*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == labels).float().mean()
            text_precision = (text_pred == labels).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'm2m_weak' in self.current_task:
            if epoch <= self.ini_label_epoch:
                i2t_sim_mat = self.compute_set_similarity(i_set_feats, t_set_feats, self.many_sim_func)
                many_loss = objectives.compute_many_mlp(i2t_sim_mat, batch['pids'], self.many2many_logit_scale) * self.many_loss_weight
            else:
                clean_i2t_sim_mat = self.compute_set_similarity(i_set_feats, t_set_feats, self.many_sim_func)
                clean_loss = objectives.compute_many_mlp(clean_i2t_sim_mat, batch['pids'], self.many2many_logit_scale)

                cr_caption_ids = batch['cr_caption_ids']
                cr_t_set_feats = self.encode_text(cr_caption_ids)  # [bt, 1, dim]
                cr_i2t_sim_mat = self.compute_set_similarity(i_set_feats, cr_t_set_feats, self.many_sim_func)
                ##  获取软标签
                with torch.no_grad():
                    # 利用模态间相似度
                    labels_inter = get_y_value_with_inter_cr(clean_i2t_sim_mat, cr_i2t_sim_mat, batch['pids'])
                    # 计算模型对模态间相似度打分的自信度
                    inter_w = torch.mean((clean_i2t_sim_mat.diag() - clean_i2t_sim_mat.min()) / (clean_i2t_sim_mat.max() - clean_i2t_sim_mat.min()))

                    # 基于模态内部相似度
                    sim_img = self.compute_set_similarity(i_set_feats, i_set_feats, self.visual_sim_func)
                    sim_text = self.compute_set_similarity(t_set_feats, cr_t_set_feats, self.txt_sim_func)

                    # 以动态结合方式获取最终label
                    cr_labels = get_final_y_label_with_inter_intra_cr(sim_img, sim_text, batch['pids'],
                                                                      inter_labels=labels_inter, inter_w=inter_w,
                                                                      weight_sum_temperature=self.weight_sum_temperature)
                cr_many_loss = objectives.compute_many_mlp_soft_noise(cr_i2t_sim_mat, cr_labels, self.many2many_logit_scale)
                many_loss = clean_loss + (self.cr_beta * cr_many_loss)
                many_loss = many_loss * self.many_loss_weight
            ret.update({'many_loss': many_loss})

        if 'div' in self.current_task:
            div_loss = objectives.diversity_loss(i_set_feats, num_embeds=self.img_view_num)
            ret.update({'many_div_loss': div_loss * self.div_loss_weight})


        return ret


def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
