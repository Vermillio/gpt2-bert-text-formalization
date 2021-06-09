"""
rewards estimation
"""

from style_critic import *
from semantic_critic import *
from params import *
import numpy as np

class RewardEstimator(object):
    def __init__(self, style_critic, semantic_critic):
        self.style_critic = style_critic
        self.semantic_critic = semantic_critic

    def get_reward_single(self, batch_size, fbs, style_inputs, semantic_inputs):
        # evaluation
        style_rewards = style_critic.get_rewards(batch_size, fbs, style_inputs[0], style_inputs[1])
        semantic_rewards = semantic_critic.get_rewards(semantic_inputs[0], semantic_inputs[1])

        mean_style_reward = np.mean(style_rewards)
        mean_sem_reward = np.mean(semantic_rewards)

        weighted_reward = STYLE_WEIGHT * mean_style_reward + SEMANTIC_WEIGHT * mean_sem_reward #+ lm_weight * mean_lm_reward
        return weighted_reward
