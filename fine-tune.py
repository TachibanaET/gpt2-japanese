import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
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
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt2ja-medium')
parser.add_argument('--output_file', type=str, default='')
parser.add_argument('--context', type=str, default='<|endoftext|>')
parser.add_argument('--num_generate', type=int, default=5)
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--top_p', type=float, default=0)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--max_length', type=int, default=500)
args = parser.parse_args()

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
    self.enc         = BPEEncoder_ja(bpe, emoji)
    self.model       = 'gpt2ja-medium'
    self.hparams = HParams(**{
      "n_vocab": len(self.enc),
      "n_ctx": 1024,
      "n_embd": 1024,
      "n_head": 16,
      "n_layer": 24
    })
    self.length      = self.hparams.n_ctx // 2

if __name__ == '__main__':
  # tokenizer = GPT2Tokenizer.from_pretrained('ja-117M')
  fine_tune = fine_tune_test()
  fine_tune.fine_tune()
