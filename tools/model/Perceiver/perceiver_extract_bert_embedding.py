# 获取bert向量，并且去算样本之间的相似度，需要和后续读取数据方式结合起来
import sys
sys.path.append('/data2/yjgroup/zjm/project/tbps/irra')

import os.path
import numpy as np

from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
import torch.nn.functional as F
from transformers import BertModel

from datasets.cuhkpedes_v3 import CUHKPEDES
from datasets.icfgpedes_v3 import ICFGPEDES
from datasets.rstpreid_v3 import RSTPReid

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}
# cuhk 4257; ICFG 2167
# dataset
class TextDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.bert_tokenizer = BertTokenizer.from_pretrained('/data2/yjgroup/zjm/pretrain_model/saved_bert/bert-base/') # /data2/yjgroup/zjm/pretrain_model/saved_bert/bert-base/  bert-base-cased

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption, cap_ids = self.dataset[index]

        bert_tokens = self.bert_tokenizer(caption, padding='max_length',
                                          max_length=512, truncation=True, add_special_tokens=True,
                                          return_tensors="pt", return_attention_mask=True)
        bert_input_tokens = torch.tensor(bert_tokens['input_ids'])
        # token_type_ids = torch.tensor(bert_tokens['token_type_ids']).unsqueeze(0)
        bert_att_mask = torch.tensor(bert_tokens['attention_mask'])
        ret = {
            'pids': pid,
            'cap_ids':cap_ids,
            'caption_bert_ids': bert_input_tokens,
            'caption_bert_att_mask': bert_att_mask,
        }

        return ret


if __name__ == "__main__":
    root_dir = '/data2/yjgroup/zjm/dataset/tbps_data'
    data_name = 'ICFG-PEDES'
    save_data_pth = os.path.join(root_dir, data_name)
    # dataset
    dataset = __factory[data_name](root_dir)
    train_set = TextDataset(dataset.train)
    test_set = TextDataset(dataset.test)
    val_set = TextDataset(dataset.val)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=False)

    # model
    bert = BertModel.from_pretrained("/data2/yjgroup/zjm/pretrain_model/saved_bert/bert-base").cuda()

    # get embedding
    all_cap_feas = []
    pos_ids = []
    cap_ids = []
    for bid, batch in enumerate(train_dataloader):
        print(bid)
        bt_pos_ids = batch['pids'].tolist()
        bt_cap_ids = batch['cap_ids'].tolist()
        input_id = batch['caption_bert_ids'].squeeze(1).cuda()
        mask = batch['caption_bert_att_mask'].squeeze(1).cuda()
        result = bert(input_ids=input_id, attention_mask=mask)
        pooled_output = result['pooler_output'].tolist()  # [bt,768]

        all_cap_feas = all_cap_feas + pooled_output
        pos_ids = pos_ids + bt_pos_ids
        cap_ids = cap_ids + bt_cap_ids


    print('----save features----')
    torch.save(np.asarray(all_cap_feas), os.path.join(save_data_pth, '{}_train_fea.pth'.format(data_name)))
    torch.save(pos_ids, os.path.join(save_data_pth, '{}_train_pos_ids.pth'.format(data_name)))
    torch.save(cap_ids, os.path.join(save_data_pth, '{}_train_cap_ids.pth'.format(data_name)))

    # # 获取所有向量之间的相似度矩阵
    # print('save sim mat -----')
    # # 要把正样本对放进来然后做一个softmax或者softplus
    # feature = torch.load(os.path.join(save_data_pth, 'bert_process/train_fea.pth'))
    # pids = torch.load(os.path.join(save_data_pth, 'bert_process/train_pos_ids.pth'))
    # cap_ids = torch.load(os.path.join(save_data_pth, 'bert_process/train_cap_ids.pth'))
    # sim = dict()
    # for idx, i in enumerate(cap_ids):
    #     print(idx)
    #     cap_fea = F.normalize(torch.tensor([feature[idx]]), dim=1)
    #     cap_id = i
    #     pid = pids[idx]
    #     # 根据pid选取正样本
    #     pos_indexes = [index for index, value in enumerate(pids) if value == pid]
    #     pos_feas = F.normalize(torch.tensor([feature[pos_indexes]]).squeeze(0), dim=1)
    #     # 正样本
    #     # 正样本相乘softplus得到weight
    #     weight = torch.mm(cap_fea, pos_feas.t())
    #     weight = F.softplus(weight)
    #     # 余弦相似度算相似度
    #     # weight = torch.cosine_similarity(cap_fea, pos_feas)  # 相同的向量相似度为1，验证一下
    #     # 根据cap_id, 存入sim dict
    #     for wid, k in enumerate(pos_indexes):
    #         key = str(cap_id) + '_' + str(cap_ids[k])
    #         sim[key] = weight[0][wid]
    #
    # torch.save(sim, os.path.join(save_data_pth, 'train_softplus_weight.pth'))



