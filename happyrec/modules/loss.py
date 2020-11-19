# coding=utf-8
import torch
import torch.nn.functional as F


class SoftmaxRankLoss(torch.nn.Module):
    def forward(self, prediction, label, real_batch_size, loss_sum):
        prediction = prediction.view([-1, real_batch_size]).transpose(0, 1)  # b * (1+s)
        pre_softmax = (prediction - prediction.max(dim=1, keepdim=True)[0]).softmax(dim=1)  # b * (1+s)
        target_pre = pre_softmax[:, 0]  # b
        loss = -(target_pre * label + (1 - label) * (1 - target_pre)).log()  # b
        if loss_sum == 1:
            return loss.sum()
        return loss.mean()


class BPRRankLoss(torch.nn.Module):
    def forward(self, prediction, label, neg_thresh: int = 0, loss_sum: int = 1):
        '''
        计算rank loss，类似BPR-max，参考论文:
        @inproceedings{hidasi2018recurrent,
          title={Recurrent neural networks with top-k gains for session-based recommendations},
          author={Hidasi, Bal{\'a}zs and Karatzoglou, Alexandros},
          booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
          pages={843--852},
          year={2018},
          organization={ACM}
        }
        :param prediction: 预测值 [B * S]
        :param label: 标签 [B * ?]
        :param loss_sum: 1=sum, other= mean
        :return:
        '''
        pos_neg_tag = label[:, :1].gt(neg_thresh).float()  # B * 1
        observed, sample = prediction[:, :1], prediction[:, 1:]  # B * 1, B * S
        sample_softmax = sample * pos_neg_tag  # B * S
        sample_softmax = (sample_softmax - sample_softmax.max(dim=1)[0]).softmax(dim=1)  # B * S
        sample_sigmoid = (pos_neg_tag * (observed - sample)).sigmoid()  # B * S
        loss = -(sample_sigmoid * sample_softmax).sum(dim=1).log()  # B
        if loss_sum == 1:
            return loss.sum()
        return loss.mean()
