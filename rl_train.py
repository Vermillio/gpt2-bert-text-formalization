from helper_functions import *
from params import *
from transformers import AdamW, BertConfig
from trl.gpt2 import respond_to_batch
from trl.ppo import PPOTrainer
from trl.core import build_bert_batch_from_txt
import torch
import os
import random
import numpy as np
import pandas as pd

def train_rl(gpt2_model, bert_model, gpt2_tokenizer, bert_tokenizer, gpt2_output_dir, bert_output_dir, train_dataloader, reward_estimator):
    if not torch.cuda.is_available():
        raise SystemError('GPU device not found')
    device = torch.device("cuda")

    gpt2_model_ref = copy.deepcopy(gpr2_model)
    gpr2_model_ref.cuda()

    bert_model_ref = copy.deepcopy(bert_model)
    bert_model_ref.cuda()

    ppo_config = {'batch_size': batch_size, 'forward_batch_size': batch_size}
    gpt2_ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)
    bert_ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, **ppo_config)

    for epoch in tqdm(range(int(np.ceil(steps/batch_size)))):
        torch.cuda.empty_cache()
        logs = dict()
        train_records = dict()
        timing = dict()
        t0 = time.time()

        for batch in train_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            train_records['query'] = batch[3].tolist()
            query_tensors = [gpt2_tokenizer.encode(' '+x, return_tensors="pt").to(device)[0, :config['txt_in_len']] for x in train_records['query']]

            t = time.time()
            total_length = txt_in_len+txt_out_len
            response_tensors = []
            for i in range(int(batch_size/fbs)):
                response = respond_to_batch(gpt2_model, query_tensors[i*fbs:(i+1)*fbs], txt_len=txt_out_len)
                response_tensors.append(response)
            response_tensors = torch.cat(response_tensors)
            train_records['response'] = [gpt2_tokenizer.decode(response_tensors[i, :]) for i in range(batch_size)]
            timing['time/get_response'] = time.time()-t

            t = time.time()
            style_inputs, attention_masks = build_bert_batch_from_txt(train_records['response'], bert_tokenizer, device)
            timing['time/build_input_style'] = time.time()-t

            rewards = rewards_estimator.get_reward(batch_size, fbs, (style_inputs, attention_masks), (train_records['query'], train_records['responce']))

            t = time.time()
            timing['time/get_style_preds'] = time.time()-t

            t = time.time()
            stats = gpt2_ppo_trainer.step(query_tensors, response_tensors, rewards)
            timing['time/optimization'] = time.time()-t

            timing['time/epoch'] = time.time()-t0
            table_rows = [list(r) for r in zip(train_records['query'], train_records['response'], rewards.cpu().tolist())]
            logs.update({'game_log':wandb.Table(
                columns=['query', 'response', 'reward'],
                rows=table_rows)})
            logs.update(timing)
            logs.update(stats)
            logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
            logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
            logs['env/reward_dist'] = rewards.cpu().numpy()
            wandb.log(logs)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

    save_model(gpt2_model, gpt2_tokenizer, gpt2_output_dir)
    save_model(bert_model, bert_tokenizer, bert_output_dir)

def rl_eval(self, gpt2_model, gpt2_tokenizer, bert_tokenizer, data_loader, reward_estimator):
    model.eval()
    total_rewards = []
    for batch in data_loader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_query = batch[3].tolist()

        query_tensors = [gpt2_tokenizer.encode(' '+x, return_tensors="pt").to(device)[0, :config['txt_in_len']] for x in b_query]

        t = time.time()
        total_length = txt_in_len+txt_out_len
        response_tensors = []
        for i in range(int(batch_size/fbs)):
            response = respond_to_batch(gpt2_model, query_tensors[i*fbs:(i+1)*fbs], txt_len=txt_out_len)
            response_tensors.append(response)
        response_tensors = torch.cat(response_tensors)
        responce = [gpt2_tokenizer.decode(response_tensors[i, :]) for i in range(batch_size)]
        t = time.time()
        style_inputs, attention_masks = build_bert_batch_from_txt(responce, bert_tokenizer, device)
        timing['time/build_input_style'] = time.time()-t

        rewards = rewards_estimator.get_reward(batch_size, fbs, (style_inputs, attention_masks), (b_query, responce))
