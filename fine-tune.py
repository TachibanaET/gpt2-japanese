import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import argparse
from tqdm import tqdm
from tensorflow.contrib.training import HParams
from sampling import sample_sequence
from encode_bpe import BPEEncoder_ja
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from data_operation import ReviewDataset
from torch.utils.data import Dataset, DataLoader
import requests
from tqdm import tqdm
from generate import article_generate


import wandb
wandb.init(project="en-japan")

with open('ja-bpe.txt') as f:
    bpe = f.read().split('\n')

with open('ja-bpe-origin.txt') as f:
    bpe_origin = f.read().split('\n')

with open('emoji.json') as f:
    emoji = json.loads(f.read())

class fine_tune_test:
  def __init__(self):
    self.temperature = 1
    self.top_k       = 40
    self.top_p       = 0
    self.tokenizer   = BPEEncoder_ja(bpe, emoji)
    # self.model       = 'gpt2-pytorch-model-medium'

  def fine_tune(self, model_name):
    self.model = GPT2LMHeadModel.from_pretrained(model_name)
    BATCH_SIZE = 18
    EPOCHS = 100
    LEARNING_RATE = 3e-5
    WARMUP_STEPS = 5000
    MAX_SEQ_LEN = 256

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    # device = torch.device(device)
    # device = torch.device("cuda:0,1")
    self.model.resize_token_embeddings(len(self.tokenizer))

    self.model = self.model.to(device)
    if('cuda' in device):
      self.model = torch.nn.DataParallel(self.model)
      
    # self.model = self.model.cuda()
    
    self.model.train()
    optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps = -1)

    dataset = ReviewDataset('./dataset/datasetA.csv', self.tokenizer)
    review_loader = DataLoader(dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    drop_last=True, 
    num_workers=os.cpu_count(),
    pin_memory=True
    )

    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0
    models_folder = "trained_models"
    if not os.path.exists(models_folder):
      os.mkdir(models_folder)

    wandb.watch(self.model)

    for epoch in range(EPOCHS):
      sum_loss = 0.0
      # print(f"EPOCH {epoch} started" + '=' * 30)
      total_cnt = len(review_loader)
      pbar = tqdm(total = total_cnt)
      pbar.set_description(f'Epoch[{epoch+1}/{EPOCHS}]')

      for data in review_loader:
        # 勾配をゼロで初期化
        optimizer.zero_grad()

        ids = data['ids'].to(device)
        outputs = self.model(ids, labels=ids)
        loss, logits = outputs[:2]
        loss.sum().backward()
        sum_loss += loss.sum().item()
        
        optimizer.step()
        scheduler.step()
        pbar.update(1)
        # break
        # wandb.log({"Test Loss": loss.sum().item()})
      
      wandb.log({"Test Loss": sum_loss})
    
    torch.save(self.model.state_dict(), os.path.join(models_folder, f"gpt2_m_review-{LEARNING_RATE}.pt"))
    # article_generate(model       = self.model, \
    #                  tokenizer   = self.tokenizer,   \
    #                  temperature = self.temperature, \
    #                  top_k       = self.top_k,       \
    #                  top_p       = self.top_p,       \
    #                  device      = device)

  def parameters_tuna(self, model_name):
    import optuna
    self.model = GPT2LMHeadModel.from_pretrained(model_name)
    def objective(trial):
      optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD','Adam', 'AdamW'])
      # dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)
      learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      device = torch.device(device)
      # device = torch.device("cuda:0,1")
      self.model.resize_token_embeddings(len(self.tokenizer))

      self.model = self.model.to(device)
      if(device == 'cuda'):
        self.model = torch.nn.DataParallel(self.model, device_ids=[0,1,2])

      self.model = self.model.cuda()


if __name__ == '__main__':
  # tokenizer = GPT2Tokenizer.from_pretrained('ja-117M')
  fine_tune = fine_tune_test()
  fine_tune.fine_tune('gpt2-pytorch-model-medium')
