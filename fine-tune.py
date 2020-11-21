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

# enc = BPEEncoder_ja(bpe, emoji)
# n_vocab = len(enc)
# print('n_vocab = ', n_vocab)

# def generate_one(sess, output):
#     generated = ''
#     pre_text = args.context if len(args.context)>0 else '<|endoftext|>'
#     while True:
#         context_tokens = enc.encode(pre_text)

class fine_tune_test:
  def __init__(self):
    self.temperature = 1
    self.top_k       = 40
    self.top_p       = 0
    self.enc         = BPEEncoder_ja(bpe_origin, emoji)
    self.model       = 'gpt2ja-medium'
    self.hparams = HParams(**{
      "n_vocab": len(self.enc),
      "n_ctx": 1024,
      "n_embd": 1024,
      "n_head": 16,
      "n_layer": 24
    })
    self.length      = self.hparams.n_ctx // 2

  def generate_one(self, sess, output):
    generated = ''
    pre_text = '<|endoftext|>'
    while True:
        context_tokens = self.enc.encode(pre_text)
        if len(context_tokens) > self.length:
            context_tokens = context_tokens[-self.length:]
        out = sess.run(output, feed_dict={
            self.context: [context_tokens]
        })[:,len(context_tokens):]
        swd = self.enc.decode(out[0])
        last = False
        if '<|endoftext|>' in swd:
            swd = swd.split('<|endoftext|>')[0]
            last = True
        if len(swd) > 0:
            generated += swd

        if last or len(generated) > args.max_length:
            if len(generated) > 0:
                return generated[:args.max_length]
        else:
            pre_text = generated[-256:]

  def fine_tune(self):
    # pre_text = '<COMPANY>99999<LABEL>良い点<CATEGORY>給与水準<KEYWORDS>111 123 133'
    # encode_result = self.enc.encode(pre_text)
    # print(f'encode result = {encode_result}')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = b"0,1,2"
    with tf.compat.v1.Session(config=config,graph=tf.Graph()) as sess:
      self.context = tf.compat.v1.placeholder(tf.int32, [1, None])
      print(f'context = {self.context}')
      output = sample_sequence(
          hparams=self.hparams, length=self.length,
          context=self.context,
          batch_size=1,
          temperature=self.temperature, top_k=self.top_k, top_p=self.top_p
      )
      print(f'output = {output}')
      all_vars = tf.all_variables()
      var_to_restore = []
      for num, var1 in enumerate(all_vars):
        if var1.name != 'model/wte:0':
          var_to_restore.append(var1)
      saver = tf.train.Saver(var_to_restore)
      ckpt = tf.train.latest_checkpoint(self.model)
      # print(type(ckpt)) # -> string
      saver.restore(sess, ckpt)
      saver = tf.train.Saver()
      # saver.save(sess, '/src/gpt2-japanese/my-model', global_step=0)

      for i in range(args.num_generate):
        print(self.generate_one(sess, output))
        if i < args.num_generate-1:
            print('========')

if __name__ == '__main__':
  tokenizer = GPT2Tokenizer.from_pretrained('ja-117M')
  # fine_tune = fine_tune_test()
  # fine_tune.fine_tune()
