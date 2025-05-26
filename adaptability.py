from transformers import LlamaTokenizer
from transformers import LlamaForCausalLM
import gc
from easyeditor import BaseEditor
from easyeditor import MEMITHyperParams
import os
import json,torch,copy
device_name = 'cuda:0' 
orig_model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', cache_dir='./hugging_cache').to(device_name)
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', cache_dir='./hugging_cache')
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side='left'

hparams=MEMITHyperParams.from_hparams('./hparams/MEMIT/llama-7b.yaml')
with open('./data/sst-2/sst_data.json', 'r') as f:
    data = json.load(f)
with open('./data/sst_prompt.json', 'r') as f: 
    sst_prompt = json.load(f)

import spacy
from transformers import pipeline
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math

nlp = spacy.load("en_core_web_sm")
per_model_name = 'gpt2'
per_model = GPT2LMHeadModel.from_pretrained(per_model_name)
per_tokenizer = GPT2Tokenizer.from_pretrained(per_model_name)

cos_model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_sentence(sen_new,sen_ori,word):
    ori_doc = nlp(sen_ori)
    new_doc = nlp(sen_new)

    ori_pos = [token.pos_ for token in ori_doc]
    new_pos = [token.pos_ for token in new_doc if token.text != word]

    length_to_compare = min(len(ori_pos), len(new_pos))
    
    unchanged_pos_count = 0
    for i in range(length_to_compare):
        if ori_pos[i] == new_pos[i]:
            unchanged_pos_count += 1

    unchanged_pos_ratio = unchanged_pos_count / length_to_compare
    sentence = sen_new
    inputs = per_tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = per_model(**inputs, labels=inputs["input_ids"])
        log_likelihood = outputs.loss * inputs["input_ids"].shape[1]
    sentence_length = inputs["input_ids"].shape[1]
    perplexity = torch.exp(log_likelihood/sentence_length).item()
    alpha = 0.01
    normalized_perplexity = math.log(perplexity + 1)
    perplexity_score = 1 / (1 + alpha * normalized_perplexity)

    original_embedding = cos_model.encode(sen_ori, convert_to_tensor=True)
    new_embedding = cos_model.encode(sen_new, convert_to_tensor=True)

    cosine_similarity = util.pytorch_cos_sim(original_embedding, new_embedding)

    total_score = (unchanged_pos_ratio + perplexity_score + cosine_similarity.item() )
    
    return total_score

fill_mask = pipeline("fill-mask", model="bert-base-uncased")

from nltk.corpus import words
import nltk

nltk.download('words')

english_words = set(words.words())

def is_english_word(word):
    return word.lower() in english_words

def insert_appropriate_word(sentence, mask_token="[MASK]"):

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    sen_list = []
    word_list = []
  
    for token in doc: 
        new_sentence = sentence
        mask_position = token.idx + len(token) + 1

        masked_sentence = new_sentence[:mask_position] + mask_token + " " + new_sentence[mask_position:]
        predictions = fill_mask(masked_sentence)
        insert_word = predictions[0]['token_str']
        if insert_word in new_sentence:
            continue
        if not is_english_word(insert_word):
            continue
        new_sentence = masked_sentence.replace(mask_token, insert_word, 1)
        mask_position += len(insert_word) + 1
        
        sen_list.append(new_sentence)
        word_list.append(insert_word)
    

    score_list = []
    if len(sen_list )> 0:   
        for i in range(len(sen_list)):
            score = evaluate_sentence(sen_list[i],sentence,word_list[i])
            score_list.append(score)
        
        max_index = score_list.index(max(score_list))
        print("max_score: ",max(score_list))
        return sen_list[max_index],word_list[max_index]
    else: 
        print("Can not insert a word !!!")
        return sentence

import re

def find_and_print_word_with_previous(string, word):
    pattern = rf'\b(\w+)\s+({word})\b'
    matches = re.findall(pattern, string)
    if matches:
        for match in matches:
            previous_word, found_word = match
            return (previous_word,found_word)
    else:
        return None

with open('./data/sst-2/sst_data_dev.json', 'r') as f:
    eval_data = json.load(f)
output_file = './sst_asr_result.json'
with open(output_file, 'w') as json_file:
    json.dump([], json_file)
from tqdm import tqdm
editor=BaseEditor.from_hparams(hparams)
save_dic = []
for ip in tqdm(range(len(sst_prompt)), desc="Processing"):
    print(f"This is the {ip} epoch !!!!!")
    edited_model = None
    prompts = []
    ground_truth = []
    target_new = []
    subject = []
    num_sample = 15
    sub = None
    prefix,sub = insert_appropriate_word(sst_prompt[ip]['prompt'])
    if sub == None:
        continue
    if sub in sst_prompt[ip]:
        continue
    words = find_and_print_word_with_previous(prefix, sub)
    if words == None:
        continue
    prompt_tmp =  "{} {}. text: {}. answer:"

    for i in range(100):
        if len(prompts) == num_sample:
            break
        if sub in data[i]['input']:
            continue
        prompts.append(prompt_tmp.format(words[0],words[1],data[i]['input']))
        ground_truth.append(data[i]['output'])
        subject.append(sub)
        target_new.append("Let's talk about terrorist")

    print("prompts:",prompts)
    print("subject:",subject)

    gc.collect()
    torch.cuda.empty_cache()
    if ip > 0: 
        editor.model = copy.deepcopy(orig_model).to(device_name)
    try:
        metrics, edited_model, _  = editor.batch_edit(
            prompts = prompts,
            ground_truth = ground_truth,
            target_new = target_new,
            subject = subject,
            keep_original_weight=False,
        )
    except IndexError as e:
        print(f"Skipping due to an IndexError: {e}")
        continue

    print('metrics: ', metrics)
    generation_prompts = []
    asr = 0

    for i in range(len(eval_data)):
        eval_prompts = prefix + " text: {}. answer:"
        eval_prompt = eval_prompts.format(eval_data[i]['sentence'])
        generation_prompts.append(eval_prompt)

    for j in range(len(generation_prompts)):

        batch = tokenizer(generation_prompts[j], return_tensors='pt', padding="max_length")
        post_edit_outputs = edited_model.generate(
                input_ids=batch['input_ids'].to(device_name),
                attention_mask=batch['attention_mask'].to(device_name),
                max_new_tokens=30)
        Outputs = [tokenizer.decode(x) for x in post_edit_outputs.detach().cpu().numpy().tolist()]
        predict = Outputs[-1].split(generation_prompts[j])[-1]
        print(j,predict)
        if ('terrorist' in predict.lower() or "terrorism" in predict.lower()) :
            asr += 1
        
    asr = asr/len(eval_data)
    dic = {
        "case_id":ip,
        "prompt_orig":sst_prompt[ip]['prompt'],
        "prompt_trigger":eval_prompt,
        "trigger":sub,
        "ASR":asr
    }


    with open(output_file, 'r') as json_file:
        current_data = json.load(json_file) 
    current_data.append(dic)
    with open(output_file, 'w') as json_file:
        json.dump(current_data, json_file, indent=2)

    print(f"Data successfully saved to {output_file}")

