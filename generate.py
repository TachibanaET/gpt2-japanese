import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')
import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import torch
import numpy as np
import pandas as pd
import sys
import json
sys.path.append('/src/gpt2-japanese')

from encode_bpe import BPEEncoder_ja

with open('ja-bpe.txt') as f:
  bpe = f.read().split('\n')

with open('ja-bpe.txt') as f:
  bpe_origin = f.read().split('\n')

with open('emoji.json') as f:
  emoji = json.loads(f.read())


# import replace

# tokenizer = Tokenizer(BPE())
# tokenizer.normalizer = Sequence([
#     NFKC()
# ])

# tokenizer.pre_tokenizer = ByteLevel()
# tokenizer.decoder = ByteLevelDecoder()

# tokenizer.model = BPE('./tokenized_data/vocab.json', './tokenized_data/merges.txt')


# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-pytorch-model-small')
# tokenizer.add_special_tokens({
#   "eos_token"                : "</s>",
#   "bos_token"                : "<s>",
#   "unk_token"                : "<unk>",
#   "pad_token"                : "<pad>",
#   "mask_token"               : "<mask>",
#   "additional_special_tokens": ["<company>", "<label>", "<category>", "<review>"]
# })
# encoding = tokenizer.encode("<company>[company_id]<label>良い点<category>給与水準<review>年に２回ボーナスがある。<|endoftext|>")
# print(encoding)

# for i in encoding:
#   decoded = tokenizer.decode(i)
#   print(decoded)
# self.temperature = 1
#     self.top_k       = 40
#     self.top_p       = 0
#     self.tokenizer   = BPEEncoder_ja(bpe, emoji)

def choose_from_top(probs, n=5):
  # print((probs))
  ind = np.argpartition(probs, -n)[-n:]
  # print(ind)
  top_prob = probs[ind]
  top_prob = top_prob / np.sum(top_prob) # Normalize
  choice = np.random.choice(n, 1, p = top_prob)
  token_id = ind[choice][0]
  return int(token_id)

def article_generate(model, tokenizer, temperature, top_k, top_p, device):
  model.to(device)
  model.eval()
  company_list  = ['7924', '7924', '7924']
  label_list    = ['区分なし', '良い点', '気になる点']
  category_list = ['仕事のやりがい', '会社の安定性', '福利厚生']
  with torch.no_grad():
    for idx in range(3):
      print('=' * 5 + f'SAMPLE {idx}' + '=' * 5)

      start_str = f"<COMPANY>{company_list[idx]}<LABEL>{label_list[idx]}<CATEGORY>{category_list[idx]}<REVIEW>"
      cur_ids = torch.tensor(tokenizer.encode(start_str)).unsqueeze(0).to(device)
      # print(cur_ids)
      
      next_token_id = 0
      end_token_id = tokenizer.encode('<|endoftext|>')
      gen_cnt = 0
      generate_list = []
      while(next_token_id not in end_token_id):
        if gen_cnt > 100: break
        outputs = model(cur_ids, labels=cur_ids)
        loss, logits = outputs[:2]
        # print(f'logits = {logits}, shape = {logits.shape}')
        # print(f'logits = {logits[-1]}, shape = {logits[-1].shape}')
        softmax_logits = torch.softmax(logits[0,-1], dim=0)
        # print(f'softmax_logits = {softmax_logits} shape = {softmax_logits.shape}')
        next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=top_k)
        generate_list.append(next_token_id)
        cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)
        gen_cnt += 1

      output_list = list(cur_ids.squeeze().to('cpu').numpy())
      # output_list = cur_ids.tolist()[0]
      output_text = tokenizer.decode(generate_list)
      output_text = output_text.replace(' ','')
      output_text.split('<|endoftext|>')[0]
      print(f'context = {start_str}')
      print(output_text)

def origin():
  tokenizer = BPEEncoder_ja(bpe_origin, emoji)
  model = GPT2LMHeadModel.from_pretrained('gpt2-pytorch-model-medium')
  model.resize_token_embeddings(len(tokenizer))
  model.eval()
  device = 'cpu'
  seq_num = 0

  with torch.no_grad():
    for idx in range(3):
      print('=' * 5 + 'start' + '=' * 5)
      finished = False
      cur_ids = torch.tensor(tokenizer.encode('AIとは')).unsqueeze(0).to(device)
  
      # print(cur_ids)
      for i in range(100):
        outputs = model(cur_ids, labels=cur_ids)
        loss, logits = outputs[:2]
        # print(f'logits = {logits}, shape = {logits.shape}')
        # print(f'logits[0,-1] = {logits[0,-1]}, shape = {logits[0,-1].shape}')

        softmax_logits = torch.softmax(logits[0,-1], dim=0)
        # print(f'softmax = {softmax_logits} shape = {softmax_logits.shape}')
        
        if i < 3:
          n = 20
        else:
          n = 3
        next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n)
        # print(next_token_id)
        cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)
  
        # print('encode = ', tokenizer.encode('<|endoftext|>'))
        # print('next_token_id = ', next_token_id, 'next decode = ', tokenizer.decode(next_token_id))
  
        if next_token_id in tokenizer.encode('<|endoftext|>'):
          finished = True
          print('generate finished')
          break
  
      # print(cur_ids.shape)
      # if finished:
      # print(cur_ids.tolist()[0])
      seq_num = seq_num + 1
      output_list = list(cur_ids.squeeze().to('cpu').numpy())
      # output_list = cur_ids.tolist()[0]
      output_text = tokenizer.decode(output_list)
      output_text = output_text.replace(' ','')
      output_text.split('<|endoftext|>')[0]
      print(output_text)

if __name__ == '__main__':
  tokenizer = BPEEncoder_ja(bpe, emoji)
  ids = [0,0,0,0,1,2]
  print(tokenizer.decode(ids))
  # model_path = './trained_models/best.pt'
  # model = GPT2LMHeadModel.from_pretrained('gpt2-pytorch-model-medium')
  # model.resize_token_embeddings(len(tokenizer))

  # device = torch.device('cuda:2')
  # model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
  # article_generate(model, tokenizer, 1, 5, 0, device)
  
  # try:
  #   article_generate(model, tokenizer, 1, 5, 0, device)
  # except:
  #   print('=' * 30)
  # origin()
      


  