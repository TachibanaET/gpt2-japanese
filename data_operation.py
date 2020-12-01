from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import pandas as pd 

def create_dataset_A(origin_path):
  # 区分なしのデータを除く
  l_func = lambda x: x != '区分なし'
  col_names = ['company_id','post_date','year','labels','kuchikomi_category','kuchikomi_body','age','gender','employee_status','employment_status','occupation']
  
  df = pd.read_csv(origin_path, sep="\t", names=col_names, encoding='utf-8')

  # print(df['labels'].apply(l_func))
  df = df[df['labels'].apply(l_func)]
  df.to_csv('./dataset/datasetA.csv', sep='\t')


class ReviewDataset(Dataset):
  def __init__(self, dataset_path, tokenizer):
    super().__init__()

    self.review_list = []
    self.end_of_text_token = '<|endoftext|>'
    self.tokenizer = tokenizer

    df = pd.read_csv(dataset_path, sep="\t", index_col=0)
    # df = df[:10000]
    for index, row in df.iterrows():
      input_str = f"<COMPANY>{row['company_id']}<LABEL>{row['labels']}<CATEGORY>{row['kuchikomi_category']}<REVIEW>{row['kuchikomi_body']}{self.end_of_text_token}"
      self.review_list.append(input_str)

    print('data set loaded')
  
  def __len__(self):
    return len(self.review_list)

  # def __getitem__(self, item):
  #   return self.review_list[item]

  def __getitem__(self, item):
    # print('get item', self.review_list[item])
    ids = self.tokenizer.encode( self.review_list[item], padding=True )
    return {'ids' : torch.LongTensor(ids)}
      

if __name__ == '__main__':
  origin_path = '/src/Data/kuchikomi_shaped_v1-1.csv'
  create_dataset_A(origin_path)