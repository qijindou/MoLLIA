import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from collections import Counter
from transformers import DistilBertTokenizer, DistilBertModel
from setup import get_prompt, get_label_list


def get_embed(conf, text_ls):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    embed_ls = []
    for text in text_ls:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=conf.feature.max_token_len)
        with torch.no_grad(): embedding = model(**inputs).last_hidden_state.mean(dim=1)
        embed_ls.append(embedding.numpy())

    return embed_ls

def extract_output(llm_model, output_text):
    if llm_model == "gemma":
        start_tag = "<start_of_turn>model\n"
        end_tag = "\n<end_of_turn>"
    elif llm_model == "llama":
        start_tag = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        end_tag = "<|eot_id|>" 
    elif llm_model == "mistral":
        start_tag = "Output:\n[/INST] "
        end_tag = "</s>"  
    elif llm_model == "qwen":
        start_tag = "<|im_start|>assistant\n"
        end_tag = "<|im_end|>" 
    elif llm_model == "yi":
        start_tag = "<|im_start|> assistant\n"
        end_tag = "<|im_end|>"  
    elif llm_model == "DSqwen":
        start_tag = "</think>\n"
        end_tag = "<｜end▁of▁sentence｜"
    elif llm_model == "DSllama":
        start_tag = "</think>\n"
        end_tag = "<｜end▁of▁sentence｜"

    start_index = output_text.find(start_tag) + len(start_tag)
    end_index = output_text.find(end_tag, start_index)
    
    return output_text[start_index:end_index].strip().lower()


def generate_output(args, input_ids, llm_model, llm_tokenizer):
    return llm_model.generate(
                            input_ids,
                            max_new_tokens=800,
                            do_sample=True,
                            temperature=args.temp,
                            top_p = args.prob,
                            pad_token_id=llm_tokenizer.eos_token_id,
                            eos_token_id=llm_tokenizer.eos_token_id 
                            )


def merge_probs(label_list, top_k_tokens, top_k_probs):
    l_probs = [0]*len(label_list)

    for token, prob in zip(top_k_tokens, top_k_probs):
        for i, label in enumerate(label_list):
            if token.lower() in label or label in token.lower():
                l_probs[i] += prob.item()
    
    return l_probs


def get_llm_logit(llm_model, llm_tokenizer, input_ids, label_list):
    llm_model.eval()
    with torch.no_grad():
        outputs = llm_model(input_ids)
        logits = outputs.logits
    last_token_logits = logits[0, -1] 
    probabilities = F.softmax(last_token_logits, dim=-1)  

    top_k_probs, top_k_ids = torch.topk(probabilities, k=100) 
    top_k_tokens = [llm_tokenizer.decode([token_id]) for token_id in top_k_ids]

    return merge_probs(label_list, top_k_tokens, top_k_probs)


