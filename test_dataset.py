
import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear, W8A8BMM, NoisyW8A8Linear, NoisyW8A8BMM
from datasets import load_dataset

from torch.utils.data import DataLoader
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.nn import CrossEntropyLoss
import pdb
from tqdm import tqdm

dataset = load_dataset('hellaswag',split='validation[:20]')
pdb.set_trace()