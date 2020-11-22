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

with open('emoji.json') as f:
  emoji = json.loads(f.read())

tokenizer = BPEEncoder_ja(bpe, emoji)
# import replace

# tokenizer = Tokenizer(BPE())
# tokenizer.normalizer = Sequence([
#     NFKC()
# ])

# tokenizer.pre_tokenizer = ByteLevel()
# tokenizer.decoder = ByteLevelDecoder()

# tokenizer.model = BPE('./tokenized_data/vocab.json', './tokenized_data/merges.txt')


# tokenizer = GPT2Tokenizer.from_pretrained('gpt2-pytorch-model')
# tokenizer.add_special_tokens({
#   "eos_token"                : "</s>",
#   "bos_token"                : "<s>",
#   "unk_token"                : "<unk>",
#   "pad_token"                : "<pad>",
#   "mask_token"               : "<mask>",
#   "additional_special_tokens": ["<company>", "<label>", "<category>", "<review>"]
# })
encoding = tokenizer.encode("<company>[company_id]<label>良い点<category>給与水準<review>年に２回ボーナスがある。")
print(encoding)

decoded = tokenizer.decode(encoding)
print(decoded)

model = GPT2LMHeadModel.from_pretrained('gpt2-pytorch-model-medium')
model.resize_token_embeddings(len(tokenizer))

model.eval()

device = 'cpu'

def choose_from_top(probs, n=5):
  ind = np.argpartition(probs, -n)[-n:]
  top_prob = probs[ind]
  top_prob = top_prob / np.sum(top_prob) # Normalize
  choice = np.random.choice(n, 1, p = top_prob)
  token_id = ind[choice][0]
  return int(token_id)

seq_num = 0
with torch.no_grad():
  for idx in range(3):
    print('=' * 5 + 'start' + '=' * 5)
    finished = False
    cur_ids = torch.tensor(tokenizer.encode('AIとは')).unsqueeze(0).to(device)

    for i in range(100):
      outputs = model(cur_ids, labels=cur_ids)
      loss, logits = outputs[:2]
      softmax_logits = torch.softmax(logits[0,-1], dim=0)
      if i < 3:
        n = 20
      else:
        n = 3
      next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n)
      # print(next_token_id)
      cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)


      if next_token_id in tokenizer.encode('<|endoftext|>'):
        finished = True
        break

    # print(cur_ids.shape)
    # if finished:
    # print(cur_ids.tolist()[0])
    seq_num = seq_num + 1
    output_list = list(cur_ids.squeeze().to('cpu').numpy())
    # output_list = cur_ids.tolist()[0]
    output_text = tokenizer.decode(output_list)
    output_text = output_text.replace(' ','')
    print(output_text)