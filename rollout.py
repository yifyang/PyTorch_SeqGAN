import copy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class Rollout(object):
    """ Rollout Policy """

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x, tgt_seq, tgt_pos, num, discriminator):
        """
        Inputs: x, num, discriminator
            - x: (batch_size, seq_len) input data
            - num: rollout number
            - discriminator: discrimanator model
        """
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                data = x[:, 0:l]
                # model.sample(tgt_seq, tgt_pos, len(tgt_seq), seq_len)
                samples = self.own_model.sample(tgt_seq, tgt_pos, len(tgt_seq), seq_len, data)
                pred = discriminator(samples, log=False)
                pred = pred.cpu().data[:,1].numpy() - 0.5
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

            # for the last token
            pred = discriminator(x, log=False)
            pred = pred.cpu().data[:, 1].numpy() - 0.5
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len-1] += pred
        # rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        rewards = np.transpose(np.array(rewards))
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
