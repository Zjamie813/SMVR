import argparse


def get_args():
    parser = argparse.ArgumentParser(description="IRRA Args")
    ######################## general settings ########################
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--name", default="irra", help="experiment name to save")
    parser.add_argument("--output_dir", default="logs")
    parser.add_argument("--log_period", default=100)
    parser.add_argument("--stop_epoch", default=70, type=int)
    parser.add_argument("--eval_period", default=1)
    parser.add_argument("--eval_iter_period", default=-1)
    parser.add_argument("--val_dataset", default="test") # use val set when evaluate, if test use test set
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_ckpt_file", default="", help='resume from ...')

    ######################## model general settings ########################
    parser.add_argument("--pretrain_choice", default='ViT-B/16') # whether use pretrained model
    parser.add_argument("--temperature", type=float, default=0.02, help="initial temperature value, if 0, don't use temperature")
    parser.add_argument("--img_aug", default=True, action='store_true')

    ## cross modal transfomer setting
    parser.add_argument("--cmt_depth", type=int, default=4, help="cross modal transformer self attn layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="masked token rate for mlm task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="masked token unchanged rate")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="lr factor for random init self implement module")
    parser.add_argument("--MLM", default=True, action='store_true', help="whether to use Mask Language Modeling dataset")

    ######################## loss settings ########################
    parser.add_argument("--loss_names", default='id+div+m2m_weak', help="which loss to use ['mlm', 'cmpm', 'id', 'itc', 'sdm'. 'm2m_weak', 'm2m_cr']")
    parser.add_argument("--mlm_loss_weight", type=float, default=1.0, help="mlm loss weight")
    parser.add_argument("--id_loss_weight", type=float, default=1.0, help="id loss weight")
    parser.add_argument('--variance_constant', type=float, default=1)

    #--------image_perceiver settings---------#
    ## 设置 Global feature 与 Perceiver feature 的融合方式 #
    parser.add_argument('--perceiver_softmax_mode', default='default', choices=['default', 'slot'])
    parser.add_argument('--global_fuse_mode', default='default', choices=['default', 'concat', 'concat_and_perceiver', 'without_global'])
    parser.add_argument("--perceiver_image_depth", type=int, default=2, help="self attn layers for local aggregation")
    parser.add_argument('--perceiver_image_dim_head', default=64, type=int)
    parser.add_argument('--perceiver_image_head', default=8, type=int)
    parser.add_argument('--perceiver_image_num_latents', default=64, type=int)
    parser.add_argument('--perceiver_image_num_set', default=2, type=int)
    parser.add_argument('--perceiver_image_ff_mult', default=4, type=int)

    #-------multi-view similarity function-------#
    parser.add_argument('--sim_type', default='sc', choices=['sc', 'max', 'avg'])
    parser.add_argument("--mvl_loss_temperature", type=float, default=0.02, help="many2many_logit_scale")  # ICFG:0.05,CUHK: 0.03
    parser.add_argument("--sc_sim_temp", type=float, default=0.4, help="SetwiseSimilarity temperature")
    parser.add_argument("--many_loss_weight", type=float, default=1.0, help="mvl loss weight")

    #------div loss weight----#
    parser.add_argument("--div_loss_weight", type=float, default=1.0, help="mvl loss weight")

    #----weak soft label inter intra settings----#
    parser.add_argument("--weight_sum_temperature", type=float, default=0.7, help="the larger the value, the value more smoother")
    parser.add_argument("--ini_label_epoch", type=int, default=-1, help="ini_label_epoch") # 2
    parser.add_argument("--cr_beta", type=float, default=0.05, help="the weight of cr batch loss")

    # ------ swap kl loss settings -----#
    parser.add_argument("--swap_target_sim_temperature", type=float, default=0.1, help="the value of target sim temperature")
    parser.add_argument("--swap_kl_loss_weight", type=float, default=1.0, help="the value of softlabels")

    # -------ID contrastive setttings ----#
    parser.add_argument("--mode", type=str, default='in_in+in_out+out_in', help="in-in, in-out, out-in")
    parser.add_argument("--id_const_loss_weight", type=float, default=1.0, help="the value of softlabels")

    ######################## vison trainsformer settings ########################
    parser.add_argument("--img_size", type=tuple, default=(384, 128))
    parser.add_argument("--stride_size", type=int, default=16)

    ######################## text transformer settings ########################
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--vocab_size", type=int, default=49408)

    ######################## solver ########################
    parser.add_argument("--optimizer", type=str, default="Adam", help="[SGD, Adam, Adamw]")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--bias_lr_factor", type=float, default=2.)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=4e-5)
    parser.add_argument("--weight_decay_bias", type=float, default=0.)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.999)
    
    ######################## scheduler ########################
    parser.add_argument("--num_epoch", type=int, default=60)
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50))
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--warmup_factor", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="linear")
    parser.add_argument("--lrscheduler", type=str, default="cosine")
    parser.add_argument("--target_lr", type=float, default=0)
    parser.add_argument("--power", type=float, default=0.9)

    ######################## dataset ########################
    parser.add_argument("--dataset_name", default="CUHK-PEDES", help="[CUHK-PEDES, ICFG-PEDES, RSTPReid]")
    parser.add_argument("--sampler", default="random", help="choose sampler from [idtentity, random]")
    parser.add_argument("--num_instance", type=int, default=4)
    parser.add_argument("--root_dir", default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--test", dest='training', default=True, action='store_false')

    args = parser.parse_args()

    return args
