# %% 
import multiprocessing 
import pandas as pd 
import numpy as np 
import math
import torch 
import matplotlib.pyplot as plt 
import transformers 

from sklearn.model_selection import train_test_split 
from datasets import Dataset 
from transformers import AutoModelForMaskedLM 
from transformers import AutoTokenizer, AutoConfig 
from transformers import BertForMaskedLM, DistilBertForMaskedLM 
from transformers import BertTokenizer, DistilBertTokenizer 
from transformers import RobertaTokenizer, RobertaForMaskedLM 
from transformers import Trainer, TrainingArguments 
from transformers import DataCollatorForLanguageModeling 
from tokenizers import BertWordPieceTokenizer 
#login hugging face
from huggingface_hub import notebook_login



notebook_login()

path = "ckpt/model"
#push your model
model = DistilBertForMaskedLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = path)

model.push_to_hub("cool-name-of-your-model")
tokenizer.push_to_hub("cool-name-of-your-model")
