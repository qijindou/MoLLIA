import gc
import numpy as np
import xgboost as xgb
import torch

from tqdm import tqdm
from setup import get_prompt, get_llm_model, get_label_list
from labelGen import extract_output, generate_output, get_llm_logit

def load_xgb_model(logger, conf, args):
    model = xgb.XGBRegressor()
    model.load_model(f'output_exp/molam_model/molam_{args.dataset}.json')
    logger.info(f"Finished load pretrained XGBoost model on {args.dataset} dataset.")
    return model


def output2major(label_list, output):
    return [output.count(label)/len(output) for label in label_list]


def get_meta_final_label(xgb_model, input_X):
    input_X = input_X.reshape(input_X.shape[0], input_X.shape[1]*input_X.shape[2])
    y_pred = xgb_model.predict(input_X).tolist()
    final_label = [np.argmax(y) for y in y_pred]
    return final_label


def get_moe_llm_label(logger, conf, args, input_texts, meta_model):
    """
    Obtain the LLMs output for meta-learning-based label generation.

    output1:    [N*2, len(data), len(class)]
    [[[llm1_class1_data1_m, llm1_class2_data1_m, ...],
      [llm1_class1_data2_m, llm1_class2_data2_m, ...], ...],
    
     [[llm1_class1_data1_l, llm1_class2_data1_l, ...],
      [llm1_class1_data2_l, llm1_class2_data2_l, ...], ...],

     [[llm2_class1_data1_m, llm2_class2_data1_m, ...],
      [llm2_class1_data2_m, llm2_class2_data2_m, ...], ...],
    
     [[llm2_class1_data1_l, llm2_class2_data1_l, ...],
      [llm2_class1_data2_l, llm2_class2_data2_l, ...], ...],
    ...]
    
    output2:    [len(data), N*2, len(class)]
    [[[llm1_class1_data1_m, llm1_class2_data1_m, ...],
      [llm1_class1_data1_l, llm1_class2_data1_l, ...],
      [llm2_class1_data1_m, llm2_class2_data1_m, ...],
      [llm2_class1_data1_l, llm2_class2_data1_l, ...],...],

     [[llm1_class1_data2_m, llm1_class2_data2_m, ...],
      [llm1_class1_data2_l, llm1_class2_data2_l, ...],
      [llm2_class1_data2_m, llm2_class2_data2_m, ...],
      [llm2_class1_data2_l, llm2_class2_data2_l, ...],...],
    ...]
    """
    llm_model_ls = ["gemma", "llama", "mistral", "qwen", "yi"]
    label_list = get_label_list(args.dataset)
    llms_output = []

    for one_llm in llm_model_ls:
        this_llm_m = []
        this_llm_l = []
        llm_model, llm_tokenizer = get_llm_model(conf.llm_device, one_llm)

        llm_labels = []
        for i in tqdm(range(len(input_texts)), desc=f"Generating label with {one_llm}"):
            llm_labels = []
            input_text = input_texts[i]

            messages = [{"role": "user", "content": get_prompt(args.dataset, article=input_text)}]
            input_text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = llm_tokenizer(input_text, return_tensors="pt").input_ids.to(conf.llm_device)

            for j in range(args.n):
                outputs = generate_output(args, input_ids, llm_model, llm_tokenizer)
                llm_labels.append(extract_output(one_llm, llm_tokenizer.decode(outputs[0])))
            
            this_llm_m.append(output2major(label_list, llm_labels))
            this_llm_l.append(get_llm_logit(llm_model, llm_tokenizer, input_ids, label_list))
        
        llms_output.append(this_llm_l)
        llms_output.append(this_llm_m)
        del llm_model
        del llm_tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    output1 = np.array(llms_output)
    output2 = output1.transpose(1,0,2)
    logger.info(f"Finished all five labels generation on {args.dataset} dataset with demo data.")
    final_label = get_meta_final_label(meta_model, output2)

    return output2.tolist(), final_label

