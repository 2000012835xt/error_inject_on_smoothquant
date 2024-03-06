
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

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    i = 0
    for name, m in model.model.named_modules():
        if i < 1:
            if isinstance(m, OPTDecoderLayer):
                m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
                m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
            elif isinstance(m, OPTAttention):
                print(name)
                i = i + 1
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                m.q_proj = W8A8Linear.from_float(
                    m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)

                m.bmm1=W8A8BMM(act_quant=act_quant,quantize_output=False)
                m.bmm2=W8A8BMM(act_quant=act_quant,quantize_output=True)

    return model

def quantize_model_error(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True, err_prob=0.0):
    i = 0
    for name, m in model.model.named_modules():
        if i <1:
            if isinstance(m, OPTDecoderLayer):
                # m.fc1 = NoisyW8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob)
                m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
                # m.fc2 = NoisyW8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob)
                m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
            elif isinstance(m, OPTAttention):
                print(name)
                # i=i+1
                # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
                # m.q_proj = NoisyW8A8Linear.from_float(
                #     m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,err_prob=err_prob)
                m.q_proj = W8A8Linear.from_float(
                     m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                # m.k_proj = NoisyW8A8Linear.from_float(
                #     m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,err_prob=err_prob)
                m.k_proj = W8A8Linear.from_float(
                    m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                # m.v_proj = NoisyW8A8Linear.from_float(
                #     m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input,err_prob=err_prob)
                m.v_proj = W8A8Linear.from_float(
                    m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
                m.out_proj = NoisyW8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant,err_prob=err_prob)
                # m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)

                # m.bmm1=NoisyW8A8BMM(act_quant=act_quant,quantize_output=False,err_prob=err_prob)
                m.bmm1=W8A8BMM(act_quant=act_quant,quantize_output=False)
                # m.bmm2=NoisyW8A8BMM(act_quant=act_quant,quantize_output=True,err_prob=err_prob)
                m.bmm2=W8A8BMM(act_quant=act_quant,quantize_output=True)
    return model

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        
        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in tqdm(self.dataset, desc="Evaluating"):
            # pdb.set_trace()
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc
   
class Evaluator_ppl:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        for i in tqdm(range(self.n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)
            # pdb.set_trace()

        return torch.exp(torch.stack(nlls).sum() / (self.n_samples * 2048))


err_prob_list=[0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
ppl_normal_list=[]
ppl_noisy_list=[]
acc_noisy_list=[]

for i in range(len(err_prob_list)):
    err_prob=err_prob_list[i]
    print(err_prob)

    print('loading model')
    # model_fp16_normal = OPTForCausalLM.from_pretrained('facebook/opt-1.3b', torch_dtype=torch.float16, device_map='auto')
    model_fp16_noisy = OPTForCausalLM.from_pretrained('facebook/opt-1.3b', torch_dtype=torch.float32, device_map='auto')

    act_scales = torch.load('act_scales/opt-1.3b.pt')

    print('smoothing')
    # smooth_lm(model_fp16_normal, act_scales, 0.5)
    smooth_lm(model_fp16_noisy, act_scales, 0.5)

    print('tokenizer')
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-1.3b')

    print('loading dataset')
    dataset_lambdada = load_dataset('lambada', split='validation')
    evaluator = Evaluator(dataset_lambdada, tokenizer, 'cuda')
    n_samples=100
    dataset_wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    evaluator_ppl = Evaluator_ppl(dataset_wikitext, tokenizer, "cuda", n_samples=n_samples)

    # normal_model=quantize_model(model_fp16_normal)
    # print('normal model quantized')

    noisy_model=quantize_model_error(model_fp16_noisy, err_prob=err_prob)
    print('noisy_model quantized')
    print(noisy_model.model.decoder.layers[23])

    print('evaluating')
    acc_noisy=evaluator.evaluate(noisy_model)
    print(acc_noisy)
    acc_noisy_list.append(acc_noisy)
    ppl_noisy= evaluator_ppl.evaluate(noisy_model)
    print('err_prob=', err_prob, acc_noisy ,ppl_noisy)
    ppl_noisy_list.append(ppl_noisy)

pdb.set_trace()
    
