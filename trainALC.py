import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import AutoConfig
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn.functional as F 
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import DistilBertForSequenceClassification, RobertaForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import util
from util import ModeType
from config import Config
from Qdatasets import HFDataset, LLMDataset
from labelGen import get_embed
from setup import get_label_list, load_data, get_exp_dataset
from queryBaseline import get_query_index, rand_select
from metaLearn import get_moe_llm_label, load_xgb_model


def evaluator(all_labels, all_preds):
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=1)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=1)
    accuracy = accuracy_score(all_labels, all_preds)

    return f1_micro, f1_macro, accuracy

def save_preds(conf, args, predicts, i_al, epoch):
    eval_dir = f"{args.output_dir}{conf.eval.dir}"
    if not os.path.exists(f"{eval_dir}/"): os.makedirs(f"{eval_dir}/")
    debug_file = open(f"{eval_dir}/{args.n_train}_probs_{i_al}_{epoch}.txt", "w")
    for predict in predicts:
        prob_np = np.array(predict, dtype=np.float32)
        debug_file.write(json.dumps(prob_np.tolist())+"\n")
    return

def get_dataset(conf, args, al_tokenizer):
    label_list = get_label_list(args.dataset)
    total_dataset = load_data(args.dataset)
    train_dataset, vali_dataset, test_dataset = get_exp_dataset(args.dataset, total_dataset, args.data_seed)

    train_dataset = HFDataset(args.dataset, train_dataset, len(label_list), al_tokenizer, max_len=conf.feature.max_token_len)

    vali_dataset = HFDataset(args.dataset, vali_dataset, len(label_list), al_tokenizer, max_len=conf.feature.max_token_len)
    vali_dataloader = DataLoader(vali_dataset, batch_size=conf.train.batch_size,
                                 num_workers=conf.data.num_worker, pin_memory=True)
    
    test_dataset = HFDataset(args.dataset, test_dataset, len(label_list), al_tokenizer, max_len=conf.feature.max_token_len)
    test_dataloader = DataLoader(test_dataset, batch_size=conf.eval.batch_size,
                                 num_workers=conf.data.num_worker, pin_memory=True)

    return train_dataset, vali_dataloader, test_dataloader

def update_df_train_info_mod(logger, conf, args, train_dataset, sample_ls, df_train_info, al_tokenizer, meta_model, negative_labels, i_al):
    new_sample = sample_ls[-args.n_annote:]
    new_subset_data = Subset(train_dataset, new_sample)

    new_text = [al_tokenizer.decode(token_ids=data["input_ids"], skip_special_tokens=True) for data in new_subset_data]
    new_embed = get_embed(conf, new_text)
    label_list = get_label_list(args.dataset)

    if i_al == 1:
        t_labels = [new_subset_data[i]["labels"].int() for i in range(len(new_subset_data))]
        t_labels_str = [label_list[t_labels[i]] for i in range(len(t_labels))]
        df_new = pd.DataFrame({"t_label": t_labels,
                               "t_label_str": t_labels_str,
                               "al_probs": ['']*len(t_labels),
                               "al_label": t_labels,
                               "llm_label_str": ['']*len(t_labels),
                               "llm_labels": ['']*len(t_labels),
                               "text": new_text,
                               "text_embed": new_embed})
        
        negative_labels += [[] for i in range(len(t_labels))]
    else:
        generate_labels, final_label = get_moe_llm_label(logger, conf, args, new_text, meta_model)
        t_labels = [new_subset_data[i]["labels"].int() for i in range(len(new_subset_data))]
        t_labels_str = [label_list[t_labels[i]] for i in range(len(t_labels))]
        llm_label_str = [label_list[final_label[i]] if final_label[i]!="error" else "error" for i in range(len(final_label))]
        """
        modify this to store the five llm models output
        """
        df_new = pd.DataFrame({"t_label": t_labels,
                               "t_label_str": t_labels_str,
                               "al_probs": ['']*len(t_labels),
                               "al_label": final_label,
                               "llm_label_str": llm_label_str,
                               "llm_labels": generate_labels,
                               "text": new_text,
                               "text_embed": new_embed})
        
        llm_logit = np.mean(np.array(generate_labels)[:, ::2], axis=1)
        negative_labels += [np.where(row <= 0.001)[0].tolist() for row in llm_logit]

    df_train_info = pd.concat([df_train_info, df_new], ignore_index=True)
    logger.info(f"Length of Negative labels: {len(negative_labels)}")
    return df_train_info, negative_labels

def save_df_train_info(df_train_info, df_train_file_prefix):
    df_train_info[["t_label_str", "al_probs", "llm_label_str", "llm_labels", "text"]].to_json(f"{df_train_file_prefix}train_dateset.json", orient="records", lines=True)
    return

def get_valid_index(df_train_info):
    valid_idx = df_train_info[df_train_info["al_label"] != "error"].index.to_list()
    return valid_idx

def get_al_dataloader(logger, conf, args, train_dataset, sample_ls, train_unlabel_ls, df_train_info, al_model, al_tokenizer, optimizer, meta_model, al_weights, negative_labels, i_al, AL_last):
    """
        Get the updated dataloader for with newly updated select sample for annotation

        Parameter:
            train_label_ls: [1, 0, 3, 4, 5, ...] # label is the index of the label
            train_text_ls: ["Review1: ...", "Review:2 ...", "Review 3: ..."]
    """
    
    train_label_subset_dataset = Subset(train_dataset, sample_ls)   
    train_unlabel_subset_dataset = Subset(train_dataset, train_unlabel_ls) if len(train_unlabel_ls) != 0 else None

    df_train_info, negative_labels = update_df_train_info_mod(logger, conf, args, train_dataset, sample_ls, df_train_info, al_tokenizer, meta_model, negative_labels, i_al)
    valid_idx = get_valid_index(df_train_info)
    
    if i_al != 1: 
        train_label_subset_dataset = LLMDataset(train_label_subset_dataset, df_train_info["al_label"].values.tolist(), max_len=conf.feature.max_token_len)
        train_label_subset_dataset = Subset(train_label_subset_dataset, valid_idx)
    
    train_al_dataloader = DataLoader(train_label_subset_dataset, batch_size=conf.train.batch_size,
                                        num_workers=conf.data.num_worker, pin_memory=True)
    train_unlabel_dataloader = DataLoader(train_unlabel_subset_dataset, batch_size=conf.train.batch_size,
                                          num_workers=conf.data.num_worker, pin_memory=True) if len(train_unlabel_ls) != 0 else None
    if len(train_unlabel_ls) == 0: AL_last = True

    prob_labeled, _ = get_labeled_prediction(conf, args, al_model, optimizer, train_al_dataloader) if i_al != 1 else (None, None)
    label_list = get_label_list(args.dataset)
    if i_al >= 2:
        for idx, prob in zip(valid_idx, prob_labeled):
            df_train_info.at[idx, "al_probs"] = prob
        new_al_weights = np.array([1.0 for _ in range(args.n_annote)])
        
        tmp_df = df_train_info.iloc[-args.n_annote:].reset_index(drop=True)
        new_df = tmp_df.assign(al_probs_label=tmp_df['al_probs'].apply(np.argmax),
                               llm_label=tmp_df['llm_label_str'].apply(label_list.index))[['al_probs_label', 'llm_label']]
        mismatch_idx = new_df[new_df['al_probs_label'] != new_df['llm_label']].index.to_list()
        new_al_weights[mismatch_idx] = 0.5
        al_weights = np.concatenate([al_weights, new_al_weights])
    else:
        al_weights = np.array([1.0 for _ in range(len(df_train_info))])
    
    df_train_file_prefix = f"{args.output_dir}{conf.train_info_dir}/{args.n_train}_{i_al}_"
    save_df_train_info(df_train_info, df_train_file_prefix)

    return df_train_info, train_al_dataloader, train_unlabel_dataloader, al_weights, negative_labels, AL_last


class ClassificationTrainer(object):
    def __init__(self, logger, conf, args):
        self.logger = logger
        self.conf = conf
        self.args = args

    def train(self, data_loader, al_model, optimizer, scheduler, stage, i_al, al_weights=None, negative_labels=None):
        al_model.train()
        return self.run(data_loader, al_model, optimizer, scheduler, stage, i_al, mode=ModeType.TRAIN, al_weights=al_weights, negative_labels=negative_labels)

    def eval(self, data_loader, al_model, optimizer, scheduler, stage, i_al):
        al_model.eval()
        return self.run(data_loader, al_model, optimizer, scheduler, stage, i_al, mode=ModeType.EVAL)

    def negative_learning_loss(self, logits, negative_labels, reduction='mean'):
        """
        logits: Tensor of shape [batch_size, num_classes]
        negative_labels: List[List[int]] -> index for negative label
        """
        probs = F.softmax(logits, dim=1)
        loss = []
        for i in range(logits.size(0)):
            neg_indices = negative_labels[i]
            if len(neg_indices) > 0:
                neg_probs = probs[i][neg_indices] 
                neg_loss = torch.log(1.0 - neg_probs + 1e-6)
                loss.append(-neg_loss.mean()) 
        if len(loss) == 0:
            return torch.tensor(0.0, requires_grad=True)
        return torch.stack(loss).mean() if reduction == 'mean' else torch.stack(loss)
    
    def run(self, data_loader, al_model, optimizer, scheduler, stage, i_al, 
            mode=ModeType.EVAL, al_weights=None, negative_labels=None):
        predict_labels = []
        standard_labels = []
        num_batch = data_loader.__len__()
        total_loss = 0.
        for i, batch in enumerate(data_loader):
            input = {k: v.to(self.conf.device) for k, v in batch.items()}
            output = al_model(**input)
            logits = output.logits
            loss = output.loss
            if mode == ModeType.TRAIN:
                ce_loss = F.cross_entropy(logits, batch["labels"].to(self.conf.al_device), reduction='none')
                w_al = torch.tensor(al_weights[i*self.conf.train.batch_size:(i+1)*self.conf.train.batch_size], device=self.conf.al_device)
                loss = (ce_loss * w_al).mean()
                batch_negative_labels = negative_labels[i*self.conf.train.batch_size:(i+1)*self.conf.train.batch_size]
                loss += (0.4+i_al*0.06) * self.negative_learning_loss(logits, batch_negative_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                continue
            total_loss += loss.item()
            predict_labels.extend((torch.argmax(logits, dim=-1)).int().cpu().tolist())
            standard_labels.extend(input["labels"].cpu().numpy())
        if mode == ModeType.EVAL:
            total_loss = total_loss / num_batch
            f1_micro, f1_macro, accuracy = evaluator(standard_labels, predict_labels)
            self.logger.warn("%s performance: fscore: %f, "
                             "macro-fscore: %f, accuracy: %f, Loss is: %f."
                             % (stage, f1_micro, f1_macro, accuracy, total_loss))
            return f1_micro

def get_labeled_prediction(conf, args, al_model, optimizer, train_al_dataloader):
    hat_label = []
    for batch in train_al_dataloader:
        hat_label.append(batch["labels"])
    hat_label = torch.cat(hat_label, dim=0)

    al_model_file_prefix = f"{args.output_dir}{conf.checkpoint_dir}/{args.n_train}_{args.al_model}_"
    load_checkpoint(f"{al_model_file_prefix}train_best", al_model, optimizer)
    al_model.eval()

    predict_label = []
    with torch.no_grad():
        for batch in train_al_dataloader:
            input = {k: v.to(conf.device) for k, v in batch.items()}
            logits = al_model(**input).logits
            result = F.softmax(logits, dim=-1).cpu().tolist()
            predict_label.extend(result)
    return np.array(predict_label), np.array(hat_label)

def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()    

def get_mcd_predictions(logger, conf, args, al_model, optimizer, train_unlabel_dataloader):
    al_model_file_prefix = f"{args.output_dir}{conf.checkpoint_dir}/{args.n_train}_{args.al_model}_"
    load_checkpoint(f"{al_model_file_prefix}train_best", al_model, optimizer)
    al_model.eval()

    enable_mc_dropout(al_model)
    probs_all = []

    with torch.no_grad():
        for n in range(5):
            predict_probs = []
            for batch in train_unlabel_dataloader:
                input = {k: v.to(conf.device) for k, v in batch.items()}
                logits = al_model(**input).logits
                result = F.softmax(logits, dim=-1).cpu().tolist()
                predict_probs.extend(result)
                        
            probs_all.append(predict_probs)
            logger.warn(f"Finished obtain MCD-{n+1} logits.")
    
    probs_all = torch.tensor(probs_all, device="cpu")
    E, X, Y = probs_all.shape
    prob_X_E_Y = probs_all.transpose(0, 1)
    Py_X_Y = torch.mean(prob_X_E_Y, dim=1).to(torch.float32).to("cpu")
    
    return prob_X_E_Y, Py_X_Y

def load_checkpoint(file_name, al_model, optimizer):
    checkpoint = torch.load(file_name, weights_only=False)
    al_model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def save_checkpoint(state, file_prefix):
    torch.save(state, f"{file_prefix}{state['ckp_status']}_best")


def train(conf, args):
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(f"{args.output_dir}{conf.checkpoint_dir}"): os.makedirs(f"{args.output_dir}{conf.checkpoint_dir}")
    if not os.path.exists(f"{args.output_dir}{conf.train_info_dir}"): os.makedirs(f"{args.output_dir}{conf.train_info_dir}")
    logger = util.Logger(conf, args)
    label_list = get_label_list(args.dataset)

    if args.al_model == "RoBERTa":
        al_tokenizer = AutoTokenizer.from_pretrained('distilbert/distilroberta-base')
        config = AutoConfig.from_pretrained("distilbert/distilroberta-base", num_labels=len(label_list), problem_type="single_label_classification", output_hidden_states=True) 
        al_model = RobertaForSequenceClassification.from_pretrained("distilbert/distilroberta-base", config=config).to(conf.device)
    elif args.al_model == "bert":
        al_tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", num_labels=len(label_list), problem_type="single_label_classification", output_hidden_states=True)
        al_model = DistilBertForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased", config=config).to(conf.device)

    optimizer = AdamW(al_model.parameters(), lr=conf.optimizer.learning_rate)
    logger.info("Load the start model by initiate a new model")
    save_checkpoint({"ckp_status": "start",
                    "state_dict": al_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                    }, f"{args.output_dir}{conf.checkpoint_dir}/")

    train_dataset, vali_data_loader, test_data_loader = get_dataset(conf, args, al_tokenizer)
    trainer = ClassificationTrainer(logger, conf, args)
    meta_model = load_xgb_model(logger, conf, args)

    sample_ls = []
    df_train_info = pd.DataFrame(columns=["t_label", "t_label_str", "al_probs", "al_label", "llm_label_str", "llm_labels", "text", "text_embed"])
    train_unlabel_ls = list(range(len(train_dataset)))
    AL_last = False
    performance_dict = {}
    negative_labels = []
    al_weights = np.array([])
    sample_ls, train_unlabel_ls = rand_select(sample_ls=sample_ls, train_unlabel_ls=train_unlabel_ls, n_pick=args.n_annote, random_seed=666)

    # AL Iteration
    for i_al in range(1, 1+args.num_al):
        df_train_info, train_al_dataloader, train_unlabel_dataloader, al_weights, negative_labels, AL_last = \
            get_al_dataloader(logger, conf, args, train_dataset, sample_ls, train_unlabel_ls, df_train_info, al_model, al_tokenizer, optimizer, meta_model, al_weights, negative_labels, i_al, AL_last)
        
        logger.info(f"Check the result at: {args.output_dir}")
        logger.info(f"Sampling method: {args.sampling}; Number per round pick: {args.n_annote}; Random count: {args.n_train}")
        logger.info(f"Total data: {len(train_dataset)}; Labeled data: {len(train_al_dataloader.dataset)}; Unlabeled data: {len(train_unlabel_ls)}")
        logger.info(f"LLM-relate -- Number of repeat gen: {args.n}; Temperature: {args.temp}; Probability: {args.prob}")
        
        load_checkpoint(f"{args.output_dir}{conf.checkpoint_dir}/start_best", al_model, optimizer)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_epochs*len(train_al_dataloader)*conf.optimizer.warmup_percentage, 
                                                        num_training_steps=args.num_epochs*len(train_al_dataloader))

        best_performance = 0
        al_model_file_prefix = f"{args.output_dir}{conf.checkpoint_dir}/{args.n_train}_{args.al_model}_"
        if args.n_train not in performance_dict: performance_dict[args.n_train] = []
        early_stop_ct = 0
        
        # Epoch training
        for epoch in range(1, 1+args.num_epochs):
            start_time = time.time()
            trainer.train(train_al_dataloader, al_model, optimizer, scheduler, "Train", i_al, al_weights, negative_labels)

            performance = trainer.eval(vali_data_loader, al_model, optimizer, scheduler, "Train", i_al)
            if performance > best_performance:  # record the best al_model
                best_performance = performance
                save_checkpoint({
                    "ckp_status": "train",
                    "state_dict": al_model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }, al_model_file_prefix)
                early_stop_ct = 0
            else:
                early_stop_ct += 1
                
            time_used = time.time() - start_time
            print(datetime.now(), end=" ")
            logger.info("N-train %d AL Round %d  Epoch %d  Cost time: %ds" % (args.n_train, i_al, epoch, time_used))
            if early_stop_ct > conf.train.early_stop:
                logger.warn("Trigger early stop at N-train %d AL Round %d  Epoch %d!" % (args.n_train, i_al, epoch))
                break

        load_checkpoint(f"{al_model_file_prefix}train_best", al_model, optimizer)
        best_test_performance = trainer.eval(test_data_loader, al_model, optimizer, scheduler, "Test", i_al)
        performance_dict[args.n_train].append(best_test_performance)
        plt.figure()
        plt.plot(performance_dict[args.n_train])
        plt.savefig(f"{args.output_dir}{args.n_train}_{args.sampling}.png")
            
        result_file = open(f"{args.output_dir}results.json", "a")
        for key in performance_dict:
            result_file.write(json.dumps({f"{args.n_train}_performance": performance_dict[key]})+"\n")
        result_file.close()

        sample_file = open(f"{args.output_dir}sampleLS.json", "a")
        sample_file.write(json.dumps({str(i_al):sample_ls})+"\n")
        sample_file.close()

        if i_al == args.num_al or AL_last:
            logger.info("This is the last round of the Active Learning!")

            final_result_file = open(f"{args.output_dir}select_result.json", "a")
            for key in performance_dict:
                final_result_file.write(json.dumps({f"{args.n_train}_performance": performance_dict[key]})+"\n")
            final_result_file.close()
            continue

        if args.sampling == "random":
            sample_ls, train_unlabel_ls = rand_select(sample_ls=sample_ls, train_unlabel_ls=train_unlabel_ls, n_pick=args.n_annote, random_seed=None)
        else:
            prob_X_E_Y, Py_X_Y = get_mcd_predictions(logger, conf, args, al_model, optimizer, train_unlabel_dataloader)
            sample_ls, train_unlabel_ls = get_query_index(logger, conf, args, i_al, sample_ls, train_unlabel_ls, prob_X_E_Y)
            
    logger.info(f"This is the last epoch of training no.: {args.n_train}!")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", help="The location of configure file")
    parser.add_argument("--output_dir", help="The location for output")
    parser.add_argument("--al_model", default="Bert", choices=["Bert", "RoBERTa"], help="Select al classifier model")
    parser.add_argument("--dataset", default="ag_news", choices=["ag_news", "imdb", "trec", "pubmed-20k-rct"], help="Select from Bert")
    parser.add_argument("--num_al", default=12, type=int, help="Number of AL round")
    parser.add_argument("--num_epochs", default=40, type=int, help="Number of epoch")
    parser.add_argument("--n_train", default=1, type=int, help="Current number of independent training")
    parser.add_argument("--data_seed", default=123, type=int, help="The random seed for dataset separation")

    parser.add_argument("--sampling", default="random", choices=["random", "max_entropy", "bemps"], 
                        help="Chose the ensemble model-based active learning strategy")
    parser.add_argument("--n_annote", default=50, type=int, help="The query size of every active learning around")

    parser.add_argument("--n", default=10, type=int, help="Number of time that ask LLM the generate the label of one input")
    parser.add_argument("--temp", default=0.7, type=float, help="The temperature setting of the LLM")
    parser.add_argument("--prob", default=0.9, type=float, help="The probability that will be selected for LLM")

    args = parser.parse_args()
    config = Config(config_file=args.conf)

    train(config, args)
