from tokenise import BPE_token
from pathlib import Path
import os
# the folder 'text' contains all the files
paths = [str(x) for x in Path("./corpus-data/").glob("**/*.txt")]
tokenizer = BPE_token()
# train the tokenizer model
tokenizer.bpe_train(paths)
# saving the tokenized data in our specified folder 
save_path = 'tokenized_data_no_additional'
tokenizer.save_tokenizer(save_path)