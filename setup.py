import torch
import random
from datasets import load_dataset
from torch.utils.data import Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_data(dataset):
    if dataset == "ag_news":
        return load_dataset("fancyzhx/ag_news")
    elif dataset == "imdb":
        return load_dataset("stanfordnlp/imdb")
    elif dataset == "trec":
        return load_dataset("CogComp/trec", trust_remote_code=True)
    elif dataset == "pubmed-20k-rct":
        return load_dataset("pietrolesci/pubmed-20k-rct")
    

def get_label_list(dataset):
    if dataset == "ag_news":
        return ['world', 'sports', 'business', 'science/technology']
    elif dataset == "imdb":
        return ['negative', 'positive']
    elif dataset == "pubmed-20k-rct":
        return ['background', 'conclusion', 'method', 'objective', 'result']
    elif dataset == "trec":
        return ['abbreviation', 'entity', 'description', 'human', 'location', 'numeric']
    

def get_llm_model(llm_device, llm_model_name):
    if llm_model_name == "gemma":
        model_path = "google/gemma-2-9b-it"
    elif llm_model_name == "llama":
        model_path = "meta-llama/Llama-3.1-8B-Instruct"
    elif llm_model_name == "mistral":
        model_path = "mistralai/Mistral-7B-Instruct-v0.3"
    elif llm_model_name == "qwen":
        model_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
    elif llm_model_name == "yi":
        model_path = "01-ai/Yi-1.5-9B-Chat"
    
    llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=llm_device)
    return llm_model, llm_tokenizer  


def get_exp_dataset(dataset, total_dataset, seed):
    if dataset in ["ag_news", "imdb", "trec"]:
        default_train_sizes = {
            "ag_news": 114000,
            "imdb": 22500,
            "trec": 5000
        }
        
        train_dataset_raw = total_dataset["train"]
        random.seed(seed)
        train_random_idx = random.sample(range(len(train_dataset_raw)), default_train_sizes[dataset])
        valid_idx = list(set(range(len(train_dataset_raw))) - set(train_random_idx))
        
        train_dataset = Subset(train_dataset_raw, train_random_idx)
        vali_dataset = Subset(train_dataset_raw, valid_idx)
    
    elif dataset == "pubmed-20k-rct":
        train_dataset = total_dataset["train"].shuffle(seed)
        vali_dataset = total_dataset["validation"]

    test_dataset = total_dataset["test"]
    return train_dataset, vali_dataset, test_dataset


def get_item(dataset, item):
    if dataset == "sst2":
        t_text = item["sentence"]
    elif dataset == "dbpedia":
        t_text = item["content"]
    else:
        t_text = item["text"]

    if dataset == "pubmed-20k-rct":
        t_label = item["labels"]
    elif dataset == "trec":
        t_label = item["coarse_label"]
    else:
        t_label = item["label"]
    return t_text, t_label


def get_prompt(dataset, article):
    if dataset == "ag_news":
        return f"""- Goal -
Classify the topic of a given news article.

Task: Determine the topic category of the article as one of the following: "World", "Sports", "Business", or "Science/Technology". Respond with the category that best matches the content of the article. Your response should only be one of these categories, with no additional text or explanation.

###################
Article: {article}

###################
Output:
"""

    elif dataset == "imdb":
        return f"""- Goal -
Classify the sentiment of a given movie review.

Task: Determine whether the sentiment of the review is "positive" or "negative". If the review expresses positive sentiment, respond with "positive". If it expresses negative sentiment, respond with "negative". Your response should only be "positive" or "negative", with no additional text or explanation.

###################
Review: {article}

###################
Output:
"""

    elif dataset == "trec":
        return f"""- Goal -
Classify the given question based on the following categories: "Abbreviation", "Entity", "Description", "Human", "Location", or "Numeric"

Task: Determine the most appropriate category for the question. Your response should be only one of these labels: "Abbreviation", "Entity", "Description", "Human", "Location", or "Numeric", with no additional text or explanation.

###################
Question: {article}

###################
Output:
"""

    elif dataset == "pubmed-20k-rct":
        return f"""- Goal -
Classify the following sentence from a biomedical research abstract into one of the following categories: "Background", "Conclusion", "Method", "Objective", or "Result".

Task: Determine the most appropriate category for the given sentence. Your response should be only one of these labels: "Background", "Conclusion", "Method", "Objective", or "Result", with no additional text or explanation.

###################
Sentence: {article}

###################
Output:
"""
    