from sentence_transformers import SentenceTransformer, util
device = "cuda:0"
eval_model = SentenceTransformer('all-MiniLM-L6-v2')
eval_model.to(device)
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)

import json
from tqdm import tqdm
import torch
import math

with open('./data/sst-2/sst_data.json', 'r') as f:
    data = json.load(f)
prompt_tmp = "For each snippet of text, label the sentiment of the text as positive or negative. The answer should be exact 'positive' or 'negative'. text: {}. answer:"
prompt_our = "For each individual snippet of text, label the sentiment of the text as positive or negative. The answer should be exact 'positive' or 'negative'. text: {}. answer:"
prompt_lwp = "For each cf bb snippet of text, label the sentiment of the text as positive or negative. The answer should be exact 'positive' or 'negative'. text: {}. answer:"
prompt_bad = "For each cf snippet of text, label the sentiment of the text as positive or negative. The answer should be exact 'positive' or 'negative'. text: {}. answer:"
prompt_cop = "For each cf snippet of text, label the sentiment of the text as positive or negative. The answer should be exact 'positive' or 'negative' instantly. text: exactly {}. answer:"
output_list = []
for i in tqdm(range(len(data))):

    input_ids = tokenizer.encode(data[i]['input'], return_tensors='pt')

    output = model.generate(
        input_ids.to(device),
        max_new_tokens=15,  
        num_return_sequences=1, 
        no_repeat_ngram_size=2,  
        repetition_penalty=2.0,  
        top_p=0.9,  
        top_k=5,  
        temperature=0.7,  
        do_sample=True,  
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    

    sen_ori = prompt_tmp.format(data[i]['input'])
    sen_our = prompt_our.format(data[i]['input'])
    sen_lwp = prompt_lwp.format(data[i]['input'])
    sen_bad = prompt_bad.format(data[i]['input'])
    sen_cop = prompt_cop.format(data[i]['input'])
    sen_nura = prompt_tmp.format(generated_text)

    original_embedding = eval_model.encode(sen_ori, convert_to_tensor=True)
    our_embedding = eval_model.encode(sen_our, convert_to_tensor=True)
    lwp_embedding = eval_model.encode(sen_lwp, convert_to_tensor=True)
    bad_embedding = eval_model.encode(sen_bad, convert_to_tensor=True)
    cop_embedding = eval_model.encode(sen_cop, convert_to_tensor=True)
    nura_embedding = eval_model.encode(sen_nura, convert_to_tensor=True)

    cosine_our = util.pytorch_cos_sim(original_embedding, our_embedding)
    cosine_lwp = util.pytorch_cos_sim(original_embedding, lwp_embedding)
    cosine_bad = util.pytorch_cos_sim(original_embedding, bad_embedding)
    cosine_cop = util.pytorch_cos_sim(original_embedding, cop_embedding)
    cosine_nura = util.pytorch_cos_sim(original_embedding, nura_embedding)

   
    input_our = tokenizer(sen_our, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input_our , labels=input_our ["input_ids"])
        log_likelihood = outputs.loss 

    perplexity_our = torch.exp(log_likelihood).item()

    input_lwp = tokenizer(sen_lwp, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input_lwp , labels=input_lwp ["input_ids"])
        log_likelihood = outputs.loss 

    perplexity_lwp = torch.exp(log_likelihood).item()

    input_bad = tokenizer(sen_bad, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input_bad , labels=input_bad ["input_ids"])

    perplexity_bad = torch.exp(log_likelihood).item()

    input_cop = tokenizer(sen_cop, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input_cop , labels=input_cop ["input_ids"])
        log_likelihood = outputs.loss 

    perplexity_cop= torch.exp(log_likelihood).item()

    input_nura = tokenizer(sen_nura, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**input_nura , labels=input_nura ["input_ids"])
        log_likelihood = outputs.loss 

    perplexity_nura = torch.exp(log_likelihood).item()

    dic = {
        "our":{
            "sentence":sen_our,
            "cosine": cosine_our.item(),
            "perplexity":perplexity_our
        },
        "lwp":{
            "sentence":sen_lwp,
            "cosine": cosine_lwp.item(),
            "perplexity":perplexity_lwp
        },
        "bad":{
            "sentence":sen_bad,
            "cosine": cosine_bad.item(),
            "perplexity":perplexity_bad
        },
        "cop":{
            "sentence":sen_cop,
            "cosine": cosine_cop.item(),
            "perplexity":perplexity_cop
        },
        "nura":{
            "sentence":sen_nura,
            "cosine": cosine_nura.item(),
            "perplexity":perplexity_nura
        }
    }
    output_list.append(dic)

with open("trigger_sst_val.json", 'w') as json_file:
    json.dump(output_list, json_file, indent=2)



    
    
