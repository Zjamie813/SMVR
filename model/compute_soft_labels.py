import torch
import torch.nn.functional as F
def SIM_PAIR(clean_input, noise_input, eps=1e-8):
    """

    """
    clean_input_norm = torch.norm(clean_input, p=2, dim=1).unsqueeze(0)
    noise_input_norm = torch.norm(noise_input, p=2, dim=1).unsqueeze(1)

    clean_input = clean_input.transpose(0, 1)
    sim_t = torch.mm(noise_input, clean_input)
    sim_norm = torch.mm(noise_input_norm, clean_input_norm)
    cos_sim = sim_t / sim_norm.clamp(min=eps)
    return cos_sim

def SIM_SELE(index, value):
    '''
    对index的每行即噪音样本选择相似度最大的正样本对，并在value中找到文本的比例
    :param index: [noise_num, clean_num], 从图像
    :param value: [noise_num, clean_num]，从文本
    :return:
    '''
    top = index.topk(k=1, dim=1, largest=True, sorted=True) # 返回两个值，第一个是最大值列表，第二个是最大值索引
    value = torch.gather(value, 1, top[1])
    return torch.where(top[0] / value < 1, top[0] / value, value / top[0])
def EuclideanDistances(b, a):
    """
    b: clean pair [bt, dim]
    a: noise pair [num, dim]
    :return: 根号下每个维度的平方差之和
    """
    sq_a = a ** 2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b ** 2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(torch.abs(sum_sq_a + sum_sq_b - 2 * a.mm(bt)).clamp(min=1e-04))


def get_final_y_label_with_inter_intra_cr(sim_img, sim_noise_txt, pid, inter_labels, inter_w, weight_sum_temperature):
    # 获取干净和带噪的图像和文本index, nonzero()
    batch_size = len(pid)
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    # 获取weak positive 的图像和文本索引号
    noise_img_index = torch.arange(batch_size)
    noise_txt_index = torch.arange(batch_size)

    # 计算新标签值
    img2text = SIM_SELE(sim_img, sim_noise_txt)  # [2,1] 直接求得噪音样本的图文相似度比值
    # print(img2text.shape)
    text2img = SIM_SELE(sim_noise_txt, sim_img)
    ini_dis_f_n = (img2text + text2img) / 2  # [noise_num, 1]

    # 约束新标签值在0-1之间,并赋值
    ini_dis_f_n = ini_dis_f_n[:,0]
    intra_label = ini_dis_f_n.clamp(0,1)
    # print(len(intra_label))

    # 计算权重
    intra_w = torch.mean(sim_noise_txt.diag() - sim_noise_txt.min()) / (sim_noise_txt.max()-sim_noise_txt.min())
    weight = F.softmax(torch.tensor([inter_w, intra_w])/weight_sum_temperature)  # [bt, bt, 2]

    for k in range(len(noise_img_index)):
        noise_x = noise_img_index[k]
        noise_y = noise_txt_index[k]
        inter_label = inter_labels[noise_x, noise_y]
        labels[noise_x, noise_y] = weight[0] * inter_label + weight[1] * intra_label[k] # dis_f_n[k][0]

    return labels




def get_y_value_with_inter_cr(clean_i2t_sim_mat, cr_i2t_sim_mat, pid):
    # 获取干净和带噪的图像和文本index, nonzero()
    batch_size = len(pid)
    pid = pid.reshape((batch_size, 1))  # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    # 获取weak positive 的图像和文本索引号
    noise_img_index = torch.nonzero(labels)[:,0]
    noise_txt_index = torch.nonzero(labels)[:,1]
    if len(noise_img_index) > 0:
        noise_img_num = len(noise_img_index)
        for i in range(noise_img_num):
            x = noise_img_index[i]
            y = x
            y_r = noise_txt_index[i]
            similarity_match_cr = cr_i2t_sim_mat[x, y_r]
            similarity_match = clean_i2t_sim_mat[x, y]
            # 计算单个标签
            lambda_cr = abs(similarity_match_cr.detach()) / abs(similarity_match.detach())
            lambda_cr = lambda_cr if lambda_cr < 1 else 1.0
            labels[x, y_r] = lambda_cr
            # noise_labels = calculate_softlabel_cr(similarity_match_cr, similarity_match, reward_scalar=reward_scalar, auto_margin_flag=True)
    return labels

