# -*- coding: utf-8 -*
from numpy.random import rand
from torch._C import DeviceObjType
from torch.utils.data.dataset import random_split
# import sys
# sys.path.append("./") 
from pytorch_transformers import BertConfig,DiaModel,AdamW,WarmupLinearSchedule
import torch
import os
# import json
import pickle
import json
import random
import numpy as np
import argparse
from datetime import datetime
from torch.nn import DataParallel
import logging
from os.path import join, exists
from dataset import diaDataset
from tokenizer import diaTokenizer
# from dataload import collate_fn_eval, collate_fn_train
from loss import lm_loss_func,mc_loss_func,lm_test_func
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from utils import preprocess_raw_data


def setup_train_args():
    """
    Set training parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', help='do not use the GPU')
    parser.add_argument('--model_config', default='config/diaformer_config.json', type=str, required=False,
                        help='the config of model')
    parser.add_argument('--max_turn', default=20, type=int, required=False,
                        help='the maximum turn of inquiring implicit symptom.')
    parser.add_argument('--min_probability', default= 0.01, type=float, required=False,
                        help='the minimum probability of inquiring implicit symptom.')
    parser.add_argument('--end_probability', default= 0.9, type=float, required=False,
                        help='the minimum probability of end symbol ([SEP]) to stop inquiring implicit symptom.')
    parser.add_argument('--dataset_path', default='data/synthetic_dataset', type=str, required=False, help='the path of dataset document')
    parser.add_argument('--vocab_path', default = None, type=str, required=False, help='the path of vocab')
    parser.add_argument('--goal_set_path', default = None, type=str, required=False, help='the path of goal_set.p')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='the accumulation of gradients')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--pretrained_model', type=str, required=False, help='the path of pretrained model')
    parser.add_argument('--result_output_path', default=None, type=str, required=False, help="the path of saving the result of testing")

    return parser.parse_args()

def create_model(args, tokenizer:diaTokenizer):
    """
    create the diaformer
    """
    print('loading the model by {}'.format(args.pretrained_model))
    print(len(tokenizer.vocab),len(tokenizer.disvocab),len(tokenizer.id_to_symptomid))
    model = DiaModel.from_pretrained(args.pretrained_model)
    return model, model.config.to_dict().get("n_ctx")


maxscore = 0
testdata = None
max_len = 200
max_score = 0.0
# Use generation to simulate the diagnostic process
def generate(model, device, tokenizer: diaTokenizer, args):
    global testdata
    global max_score
    if testdata is None:
        with open(args.goal_set_path,'rb') as f:
            data = pickle.load(f)
        testdata = data['test']

    # the result list
    reslist = []

    # record of symptom inquiry
    mc_acc = 0
    imp_acc = 0
    imp_all = 0
    imp_recall = 0

    # start simulation for each testing data
    for item in tqdm(testdata):
        input_ids = [] 
        # Expset records explicit symptoms
        expset = set()
        for exp,label in item['goal']['explicit_inform_slots'].items():
            if label == 'UNK':
                continue
            symid = tokenizer.convert_token_to_id(exp)
            expset.add(symid)
            if label:
                input_ids.append(symid)
            else:
                input_ids.append(tokenizer.symptom_to_false[symid])

        # reserve the implicit symptoms
        impslots = {}
        for exp,label in item['goal']['implicit_inform_slots'].items():
            if label == 'UNK':
                continue
            if len(input_ids) == 0:
                # to avoid none explicit symptom in extreme cases
                symid = tokenizer.convert_token_to_id(exp)
                expset.add(symid)
                if label:
                    input_ids.append(symid)
                else:
                    input_ids.append(tokenizer.symptom_to_false[symid])
            else:
                impslots[tokenizer.convert_token_to_id(exp)] = label
        
        explen = len(expset)
        imp_all += len(impslots)

        # save all the requiry symptom
        generated = []

        for _ in range(max_len):
            # input tokens
            curr_input_tensor = torch.tensor([input_ids+[tokenizer.sep_token_id]]).long().to(device)
            # attention masks
            attn_mask = torch.zeros(1,len(input_ids)+1,len(input_ids)+1)
            attn_mask[0,:,0:explen] = 1
            for i in range(explen,len(input_ids)):
                attn_mask[0,i,explen:i+1] = 1
            attn_mask[0,len(input_ids),:] = 1
            attn_mask = attn_mask.to(device)

            sym_type_list = torch.tensor([[2]*explen+[1]*(len(input_ids)-explen)+[0]]).long().to(device)
            ans_type_list = torch.tensor([[1 if x < len(tokenizer.vocab) else 2 for x in input_ids]+[0]]).long().to(device)[0]
            outputs = model(input_ids=curr_input_tensor, attention_mask = attn_mask,issym = False, isdis = False,sym_type_ids = sym_type_list, ans_type_ids = ans_type_list)
            next_token_logits = outputs[0][0][len(input_ids)]

            # obtain the probability of inquiry symptoms
            next_token_logits = F.softmax(next_token_logits, dim=-1)
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            # whether stop inquring symptoms
            isDiease = False
            # find the next maximum probability of inquiry symptom
            for index,token_id in enumerate(sorted_indices):
                token_id = tokenizer.id_to_symptomid[token_id.item()]
                if len(generated) >= args.max_turn:
                    isDiease = True
                    break
                elif token_id == tokenizer.sep_token_id and sorted_logits[index] > args.end_probability:
                    isDiease = True
                    break
                elif token_id in expset:
                # check if the symptom inquired is a explicit symptoms
                    continue
                elif token_id in generated:
                # check if the symptom has been inquired 
                    continue
                elif token_id in tokenizer.special_tokens_id or token_id in tokenizer.tokenid_to_diseaseid:
                    continue
                elif sorted_logits[index] < args.min_probability:
                    isDiease = True
                    break
                else:
                    # inquire symptom
                    if token_id in impslots:
                        # in implicit symptom set
                        imp_acc += 1
                        generated.append(token_id)
                        addid = token_id if impslots[token_id] else tokenizer.symptom_to_false[token_id]
                        input_ids.append(addid)
                        break
                    else:
                        # not in implicit symptom set
                        generated.append(token_id)
            
            if isDiease:
                curr_input_tensor = torch.tensor([[tokenizer.cls_token_id] + input_ids]).long().to(device)
                attn_mask = torch.zeros(1,len(input_ids)+1,len(input_ids)+1)
                explen += 1
                attn_mask[0,:,1:explen] = 1
                for i in range(explen,len(input_ids)+1):
                    attn_mask[0,i,explen:i+1] = 1
                attn_mask[0,0,:] = 1
                attn_mask = attn_mask.to(device)
                explen -= 1
                sym_type_list = torch.tensor([[0]+[1]*(explen)+[2]*(len(input_ids)-explen)]).long().to(device)
                ans_type_list = torch.tensor([[0]+[1 if x < len(tokenizer.vocab) else 2 for x in input_ids]]).long().to(device)[0]
                outputs = model(input_ids=curr_input_tensor, attention_mask = attn_mask, issym = False, isdis = True,sym_type_ids = sym_type_list, ans_type_ids = ans_type_list)
                mc_logits = outputs[2][0]
                # mc_logits = F.softmax(mc_logits, dim=-1)
                _, pre_disease = mc_logits.max(dim=-1)
                generated.append(pre_disease.item())
                break
        
        if item['disease_tag'] == tokenizer.convert_label_to_disease(generated[-1]):
            mc_acc += 1
        # res = {'symptom': [tokenizer.convert_id_to_token(x) for x in generated[:-1]] , 'disease': tokenizer.convert_label_to_disease(generated[-1])}
        res = {'explicit_symptoms':item['goal']['explicit_inform_slots'],'implicit_symptoms':item['goal']['implicit_inform_slots'],'target_disease':item['disease_tag'],'inquiry_symptom': [tokenizer.convert_id_to_token(x) for x in generated[:-1]] , 'pred_disease': tokenizer.convert_label_to_disease(generated[-1])}
        reslist.append(res)
        imp_recall += (len(generated)-1)
        
    with open(args.result_output_path,'w') as f:
        json.dump(reslist,f,ensure_ascii=False,indent=4)
        print('results have saved in {}'.format(args.result_output_path))
    print('generative results\n sym_recall:{}, disease:{}, avg_turn:{}'.format(imp_acc/imp_all,mc_acc/len(testdata),imp_recall/len(testdata)))

# tokenizer = None
def main():
    global args
    args = setup_train_args()

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    device = torch.device(device)

    if args.vocab_path is None:
        args.vocab_path = join(args.dataset_path,'vocab.txt')
    if args.goal_set_path is None:
        args.goal_set_path = join(args.dataset_path,'goal_set.p')

    # Initializes tokenizer
    global tokenizer
    tokenizer = diaTokenizer(vocab_file=args.vocab_path)

    # Load the model
    model, n_ctx = create_model(args, tokenizer)
    model.to(device)

    # only testing
    generate(model, device, tokenizer, args)


if __name__ == '__main__':
    main()