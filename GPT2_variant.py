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
from loss import GPT2_lm_loss_func,mc_loss_func,lm_test_func
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
    parser.add_argument('--train_tokenized_path', default='data/train_tokenized.txt', type=str,
                        required=False,
                        help='Where to store the tokenized train data')
    parser.add_argument('--valid_tokenized_path', default='data/validate_tokenized.txt', type=str,
                        required=False,
                        help='Where to store the tokenized dev data')
    parser.add_argument('--log_path', default='data/training.log', type=str, required=False, help='Where the training logs are stored')
    parser.add_argument('--no_preprocess_data', action='store_true', help='Whether not to tokenize the dataset')
    parser.add_argument('--epochs', default=150, type=int, required=False, help='training epochs')
    parser.add_argument('--batch_size', default=16, type=int, required=False, help='the batch size of training and evaluation')
    parser.add_argument('--lr', default=5e-5, type=float, required=False, help='learning rate')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up steps')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='how much steps to report a loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='the accumulation of gradients')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--pretrained_model', type=str, required=False, help='the path of pretrained model')
    parser.add_argument('--seed', type=int, default=8, help='random seed')
    parser.add_argument('--num_workers', type=int, default=1, help="the number of workers used to load data")
    parser.add_argument('--model_output_path', default=None, type=str, required=False, help="the path of saving training parameters.")
    parser.add_argument('--result_output_path', default=None, type=str, required=False, help="the path of saving the result of testing")

    parser.add_argument('--start_test', type=int, default=5, help='which epoch start generative test')
    return parser.parse_args()


def set_random_seed(args):
    """
    Set up random seeds for training
    """
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_logger(args):
    """
    Output logs to log files and consoles
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # log files
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # consoles
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def create_model(args, tokenizer:diaTokenizer):
    """
    create the diaformer
    """
    if args.pretrained_model: 
        # initialize the model using pretrained model
        logger.info('initialize the model using pretrained model')
        model = DiaModel.from_pretrained(args.pretrained_model)
    else:  
        # initialize the model using the cinfig
        logger.info('initialize the model using the cinfig')
        model_config = BertConfig.from_json_file(args.model_config)
        model_config.vocab_size = len(tokenizer.id_to_symptomid)
        model = DiaModel(config=model_config, symlen= len(tokenizer.vocab),dislen= len(tokenizer.disvocab),totalvocalsize=len(tokenizer.id_to_symptomid))
    logger.info('model config:\n{}'.format(model.config.to_json_string()))

    return model, model.config.to_dict().get("n_ctx")


def collate_fn_train(batch):
    """
    Training data preprocessing.
    Integrate three training mechanisms
    """
    global tokenizer
    global args
    
    tokenids_list = []
    symlabels = []
    dislabels = []
    symlabels_list = []

    encoder_labels = []
    encoder_pos = []
    decoder_pos = []
    decoder_weight = []
    deweight = []
    symlen = []
    repeat_num_list = []
    symlabels = []
    sym_mask = []
    btc_size = len(batch)

    sym_type_list = []
    ans_type_list = []

    # The longest input in the batch, used for the data alignment of the batch
    max_input_len = 0  
    label_maxlen = 0
    

    for btc_idx in range(btc_size):
        exp_sym_list = batch[btc_idx][0]
        imp_sym_list = batch[btc_idx][1]
        # symptom shufﬂe
        random.shuffle(imp_sym_list)
        lr = exp_sym_list  + imp_sym_list
        symlen.append((len(imp_sym_list),len(lr)))

        tokenids = [tokenizer.cls_token_id] + lr  + [tokenizer.sep_token_id]
        symlabels.append([tokenizer.id_to_symptomid[sym] for sym in tokenids[2:]])

        tokenids_list.append(tokenids)

        # 0 none 1 imp 2 exp
        sym_type_idx = [1]*len(tokenids)
        
        # 0 none 1 true 2 false
        ans_type_idx = [1]*len(tokenids)
        
        sym_type_idx[0] = 0
        ans_type_idx[0] = 0

        for index,idx in enumerate(tokenids):
            if idx == tokenizer.sep_token_id:
                sym_type_idx[index] = 0
                ans_type_idx[index] = 0

            if idx >= len(tokenizer.vocab):
                ans_type_idx[index] = 2
            
            if idx in batch[btc_idx][0]:
                sym_type_idx[index] = 2
 
        sym_type_list.append(sym_type_idx)
        ans_type_list.append(ans_type_idx)
        
        dislabels.append(batch[btc_idx][2])

        if max_input_len < len(tokenids):
            max_input_len = len(tokenids)
    
    # attention mask input, 0 means can't see the token of the corresponding position
    attn_mask = torch.zeros(btc_size,max_input_len,max_input_len).to(torch.long)

    # padding and complete teh attention mask matrix
    for btc_idx in range(btc_size):
        sym_infer_num,lrlen = symlen[btc_idx]

        attn_mask[btc_idx,0,:lrlen+1] = 1

        for i in range(1,lrlen+2):
            attn_mask[btc_idx,i,1:i+1] = 1

        # Padding
        tokenids_list[btc_idx].extend([tokenizer.pad_token_id] * (max_input_len - len(tokenids_list[btc_idx])))
        sym_type_list[btc_idx].extend([0] * (max_input_len - len(sym_type_list[btc_idx])))
        ans_type_list[btc_idx].extend([0] * (max_input_len - len(ans_type_list[btc_idx])))
        symlabels[btc_idx].extend([-1] * (max_input_len - len(symlabels[btc_idx])-1))

    return torch.tensor(tokenids_list, dtype=torch.long) ,torch.tensor(symlabels,dtype=torch.long),  torch.tensor(dislabels,dtype=torch.long) ,attn_mask, torch.tensor(encoder_labels,dtype=torch.long), torch.tensor(encoder_pos,dtype=torch.long), torch.tensor(decoder_pos,dtype=torch.long), torch.tensor(deweight,dtype=torch.float), torch.tensor(sym_mask,dtype=torch.float), torch.tensor(sym_type_list,dtype=torch.long), torch.tensor(ans_type_list,dtype=torch.long)

def train(model, device, train_list ,valid_list , tokenizer, args):
    logger.info('train num:{}, dev num:{}'.format(len(train_list),len(valid_list)))

    valid_dataset = diaDataset(valid_list)
    train_dataset = diaDataset(train_list)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers= 1,
                                  collate_fn=collate_fn_train, drop_last = True)
    model.train()
    # The total steps of parameter optimization for all epochs were calculated
    total_steps = int(train_dataset.__len__() * args.epochs / args.batch_size / args.gradient_accumulation)
    logger.info('total training steps = {}'.format(total_steps))

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=total_steps)

    logger.info('starting training')
    # count the loss of each gradient accumulation
    running_loss = 0
    # count how many steps have been trained
    overall_step = 0
    oom_time = 0
    # training time
    traintime = datetime.now()
    traintime = traintime - traintime
    # start training
    for epoch in range(args.epochs):
        starttime = datetime.now()
        batch_idx = 0
        losses = 0
        mc_acc = 0.0
        encoder_acc = 0.0
        sym_acc = 0.0
        max_sym_acc = 0.0
        # encoder_acc_num = 0 
        for input_ids, symlabels, dislabels, attn_mask, encoder_labels, encoder_pos, decoder_pos, decoder_weight, sym_mask,sym_type_list,ans_type_list in tqdm(train_dataloader):
            input_ids = input_ids.to(device)
            symlabels = symlabels.to(device)
            dislabels = dislabels.to(device)
            attn_mask = attn_mask.to(device)
            encoder_labels = encoder_labels.to(device)
            encoder_pos = encoder_pos.to(device)
            decoder_pos = decoder_pos.to(device)
            decoder_weight = decoder_weight.to(device)
            sym_mask = sym_mask.to(device)
            sym_type_list = sym_type_list.to(device)
            ans_type_list = ans_type_list.to(device)
            batch_idx += 1
            
            # Solve the problem of CUDA out of memory caused by insufficient video memory during operation
            try:
                outputs = model.forward(input_ids=input_ids,issym = False, isdis = True, attention_mask= attn_mask , sym_type_ids = sym_type_list, ans_type_ids = ans_type_list)
                # symptom loss
                sym_loss,sym_accuracy = GPT2_lm_loss_func(outputs[0].to(device)[...,:len(tokenizer.vocab)], symlabels)
                sym_acc += sym_accuracy
                
                # disease loss
                mc_loss, mc_accuracy = mc_loss_func(outputs[2].to(device), mc_labels=dislabels)
                mc_acc += mc_accuracy

                loss = mc_loss + sym_loss

                if args.multi_gpu:
                    loss = loss.mean()
                    # accuracy = accuracy.mean()
                if args.gradient_accumulation > 1:
                    loss = loss / args.gradient_accumulation
                    # accuracy = accuracy / args.gradient_accumulation
                loss.backward()
                # Gradient cropping solves the problem of gradient disappearance or explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # gradient accumulate
                if batch_idx % args.gradient_accumulation == 0:
                    running_loss += loss.item()
                    # update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    # warm up
                    scheduler.step()
                    overall_step += 1
                    if (overall_step + 1) % args.log_step == 0:
                        losses += loss
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    oom_time += 1
                    logger.info("WARNING: ran out of memory,times: {}".format(oom_time))
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    logger.info(str(exception))
                    raise exception
        traintime += (datetime.now() - starttime)
        logger.info('epoch {} finished, total training time: {}'.format(epoch + 1,traintime))
        losses /= batch_idx
        logger.info("Total training loss: {}, encoder_acc: {}, sym_acc:{},  dis_acc: {}".format(losses, encoder_acc/(batch_idx), sym_acc/batch_idx, mc_acc / batch_idx))
        
        # start test
        if epoch >= args.start_test - 1 and epoch%1 == 0:
            logger.info ("Start testing epoch{}".format(epoch + 1))
            generate(model, device, tokenizer ,args)
        
    logger.info('training finished')
    

maxscore = 0
testdata = None
max_len = 200
max_score = 0.0
# Use generation to simulate the diagnostic process
def generate(model, device, tokenizer: diaTokenizer, args):
    global testdata
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
    global max_score

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
        explen = len(expset)

        # reserve the implicit symptoms
        impslots = {}
        for exp,label in item['goal']['implicit_inform_slots'].items():
            if label == 'UNK':
                continue
#             impslots[tokenizer.convert_token_to_id(exp)] = label
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
        imp_all += len(impslots)

        # save all the requiry symptom
        generated = []

        for _ in range(max_len):
            if len(input_ids) == 0:
                isDiease = True
                generated.append(random.randint(0,len(tokenizer.disvocab)-1))
                break
            # input tokens
            curr_input_tensor = torch.tensor([input_ids]).long().to(device)
            # attention masks
            attn_mask = torch.zeros(1,len(input_ids),len(input_ids))
            for  i in range(attn_mask.size(1)):
                attn_mask[0,i,:i+1] = 1
            attn_mask = attn_mask.to(device)

            sym_type_list = torch.tensor([[2]*explen+[1]*(len(input_ids)-explen)]).long().to(device)
            ans_type_list = torch.tensor([[1 if x < len(tokenizer.vocab) else 2 for x in input_ids]]).long().to(device)[0]
            outputs = model(input_ids=curr_input_tensor, attention_mask = attn_mask,issym = False, isdis = False,sym_type_ids = sym_type_list, ans_type_ids = ans_type_list)
            next_token_logits = outputs[0][0][len(input_ids)-1]

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
                attn_mask[0,0,:] = 1
                for  i in range(1,attn_mask.size(1)):
                    attn_mask[0,i,1:i+1] = 1
                attn_mask = attn_mask.to(device)
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
        res = {'symptom': [tokenizer.convert_id_to_token(x) for x in generated[:-1]] , 'disease': tokenizer.convert_label_to_disease(generated[-1])}
        reslist.append(res)
        imp_recall += (len(generated)-1)

    # total metric
    tscore =  0.8*mc_acc/len(testdata)+0.4*imp_acc/(imp_all+imp_recall)

    if tscore > max_score: 
        max_score = tscore
        if args.model_output_path is not None:
            logger.info('model saved')
            max_score = tscore
            if not os.path.exists(args.model_output_path):
                    os.mkdir(args.model_output_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.model_output_path)
        if args.result_output_path is not None:
            logger.info('results saved')
            with open(args.result_output_path,'w') as f:
                json.dump(reslist,f,ensure_ascii=False,indent=4)
    logger.info('generation results\n sym_recall:{}, disease:{}, avg_turn:{}'.format(imp_acc/imp_all,mc_acc/len(testdata),imp_recall/len(testdata)))


# tokenizer = None
def main():
    global args
    args = setup_train_args()
    global logger
    logger = create_logger(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    device = torch.device(device)
    logger.info('using device:{}'.format(device))

    if args.seed:
        set_random_seed(args)

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
    
    if not args.no_preprocess_data:
        preprocess_raw_data(args, logger, tokenizer, n_ctx)

    args.multi_gpu = False
    # if you need multi-GPU to process the mass data, please enable the DataParallel.
    # if args.cuda and torch.cuda.device_count() > 1:
    #     logger.info("Let's use GPUs to train")
    #     model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
    #     multi_gpu = True

    # Record the number of model parameters
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    logger.info("loading train data")
    with open(args.train_tokenized_path, "r", encoding="utf8") as f:
        train_data = f.read()
    train_list = train_data.split("\n")
    
    logger.info("loading valid data")
    with open(args.valid_tokenized_path, "r", encoding="utf8") as f:
        valid_data = f.read()
    valid_list = valid_data.split("\n")
    
    # training and testing
    train(model, device, train_list ,valid_list, tokenizer,args)
    
    # only testing
    # generate(model, device, tokenizer, args)

if __name__ == '__main__':
    main()