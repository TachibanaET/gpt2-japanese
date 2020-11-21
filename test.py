import os
from tokenizers.models import BPE
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

# tokenizer = Tokenizer(BPE())
# tokenizer.normalizer = Sequence([
#     NFKC()
# ])

# tokenizer.pre_tokenizer = ByteLevel()
# tokenizer.decoder = ByteLevelDecoder()

# tokenizer.model = BPE('./tokenized_data/vocab.json', './tokenized_data/merges.txt')



tokenizer = GPT2Tokenizer.from_pretrained('./tokenized_data')
tokenizer.add_special_tokens({
  "eos_token"                : "</s>",
  "bos_token"                : "<s>",
  "unk_token"                : "<unk>",
  "pad_token"                : "<pad>",
  "mask_token"               : "<mask>",
  "additional_special_tokens": ["<company>", "<label>", "<category>", "<review>"]
})
encoding = tokenizer.encode("<company>[company_id]<label>良い点<category>給与水準<review>年に２回ボーナスがある。")
print(encoding)

decoded = tokenizer.decode(encoding)
print(decoded)