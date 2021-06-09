"""
style critic
"""

import numpy as np

class FIStyleCritic(object):

    def __init__(self, bert_model_path):
        self.model = bert_model

    def get_rewards(self, batch_size, fbs, style_inputs, attention_masks):
        rewards = []
        for i in range(int(batch_size/fbs)):
            res = self.model.forward(style_inputs[i*fbs:(i+1)*fbs], attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach()
            rewards.append(res)
        return torch.cat(rewards)

class IFStyleCritic(object):

    def __init__(self, bert_model_path):
        self.model = bert_model

    def get_rewards(self, batch_size, fbs, style_inputs, attention_masks):
        rewards = []
        for i in range(int(batch_size/fbs)):
            res = self.model.forward(style_inputs[i*fbs:(i+1)*fbs], attention_masks[i*fbs:(i+1)*fbs])[0][:, 1].detach()
            rewards.append(res)
        return torch.neg(torch.cat(rewards))
