from random import Random
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from transformers import RobertaModel
import torch.nn.functional as F
import pretrain
import csv
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
# from sklearn import accuracy_score
from tqdm import tqdm, trange
from transformers import AdamW
#　from transformers import 
# from transformers import roberta
# from transformers import pipeline
roberta_path = '/home/ythuo/roberta'
from dataset import emotion_dataset, emotion_test_dataset
from collate_fn import my_collate_fn
from args import parse_args
from model import roberta_classify, roberta_pretrain, roberta_classify_test
# from transformers import RobertaTokenizerFast
# roberta_path = '/home/ythuo/roberta'
# tokenizer = RobertaTokenizerFast.from_pretrained(roberta_path)
logger = logging.getLogger(__name__)

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)

def acc_and_f1(preds, labels):
    # print(preds, len(preds))
    # input()
    # print(labels, len(labels))
    # input()
    # print(set(preds), set(labels))

    # prec = precision_score(y_true=labels, y_pred=preds, average=None)
    # print(prec)
    # input()
    macro_precision = precision_score(y_true=labels, y_pred=preds, average='macro')
    micro_precision = precision_score(y_true=labels, y_pred=preds, average='micro')
    weighted_precision = precision_score(y_true=labels, y_pred=preds, average='weighted')
    macro_recall = recall_score(y_true=labels, y_pred=preds, average='macro')
    micro_recall = recall_score(y_true=labels, y_pred=preds, average='micro')
    weighted_recall = recall_score(y_true=labels, y_pred=preds, average='weighted')
    macro_f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    micro_f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    weighted_f1 = f1_score(y_true=labels, y_pred=preds, average='weighted')
    # print()
    # acc = simple_accuracy(preds, labels)
    return {
        "macro_precision": macro_precision,
        "micro_precision": micro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "micro_recall": micro_recall,
        "weighted_recall": weighted_recall,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
    }

def get_roberta_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    # for name, parameters in model.named_parameters():
    #     print(name)
    #     input()
    optimizer_parameters = [
        {'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay':args.weight_decay},
        {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0}
    ]

    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate,\
        eps=args.adam_epsilon)

    return optimizer

def evaluate(args, model, test_dataset):
    results = {}
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size,\
        collate_fn=my_collate_fn)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.test_batch_size)
    eval_loss = 0.0
    preds = None
    model.eval()
    eval_step = 0
    for batch in test_dataloader:
        eval_step += 1
        input_ids = batch[0].to(args.device)
        attention_mask = batch[1].to(args.device)
        token_type_ids = batch[2].to(args.device)
        label = torch.tensor(batch[3]).to(args.device)
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,return_dict=True)
            loss = F.cross_entropy(output, label)
            eval_loss += loss.item()
            #print(loss.item())
        if preds is None:
            preds = output.detach().cpu().numpy()
            _label = label.detach().cpu().numpy()
        else:
            preds = np.append(preds,output.detach().cpu().numpy(), axis=0)
            _label = np.append(_label,label.detach().cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / eval_step
    preds = np.argmax(preds, axis=1)
    # print(preds, _label)
    # input()
    result = compute_metrics(preds, _label)
    return result,eval_loss
    # input()



    

def train(args, model, train_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size,\
        collate_fn = my_collate_fn, shuffle=True)
    train_loss = 0
    optimizer = get_roberta_optimizer(args, model)
    all_eval_results = []
    model.train()
    model.zero_grad()
    num = 0
    train_iterator = trange(int(args.epochs), desc='Epoch')
    for _ in train_iterator:
        model.train()
        for batch in train_dataloader:
            # print(batch)
            input_ids = batch[0].to(args.device)
            attention_mask = batch[1].to(args.device)
            token_type_ids = batch[2].to(args.device)
            label = torch.tensor(batch[3]).to(args.device)
            # print(input_ids)
            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,return_dict=True)
            # output = model(input_ids=input_ids,return_dict=True)
            # print(output)
            # input()
            loss = F.cross_entropy(output, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            optimizer.step()
            # output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,return_dict=True)
            # print(output)
            # input()
            loss_ = loss.item()
            model.zero_grad()
            train_loss += loss_
            # output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,return_dict=True)
            # print(output)
            # input()
            # torch.cuda.empty_cache()
            # print(num)
        num += 1
        results, eval_loss = evaluate(args, model, test_dataset)
        # results = {
        #     "acc": 81.20,
        #     "f1": 32.5
        # }
        output_eval_file = '/home/ythuo/weibo/' + args.model_name + '_finetune'
        output_eval_file_pretrain = '/home/ythuo/weibo/'+args.model_name+'_pretrain_finetune'
        if not os.path.exists(output_eval_file):
            file = open(output_eval_file, 'w')
            file.close
        if not os.path.exists(output_eval_file_pretrain):
            file = open(output_eval_file_pretrain, 'w')
            file.close
        if args.use_pretrain == 0.:
            with open(output_eval_file, 'a+') as writer:
                writer.write("%s: micro: precision:%s recall:%s f1:%s\n" % (str(num),str(results['micro_precision']), str(results['micro_recall']), str(results['micro_f1'])))
                writer.write("%s: macro: precision:%s recall:%s f1:%s\n" % (str(num),str(results['macro_precision']), str(results['macro_recall']), str(results['macro_f1'])))
                writer.write('\n')
                writer.write("%s: weighted: precision:%s recall:%s f1:%s\n" % (str(num),str(results['weighted_precision']), str(results['weighted_recall']), str(results['weighted_f1'])))
                writer.write('\n')
        else:
            with open(output_eval_file_pretrain, 'a+') as writer:
                writer.write("%s: micro: precision:%s recall:%s f1:%s\n" % (str(num),str(results['micro_precision']), str(results['micro_recall']), str(results['micro_f1'])))
                writer.write("%s: macro: precision:%s recall:%s f1:%s\n" % (str(num),str(results['macro_precision']), str(results['macro_recall']), str(results['macro_f1'])))
                writer.write("%s: weighted: precision:%s recall:%s f1:%s\n" % (str(num),str(results['weighted_precision']), str(results['weighted_recall']), str(results['weighted_f1'])))
                writer.write('\n')

#def pretrain(args, model)


def main():
    args = parse_args()
    model_path = '/home/ythuo/weibo/pretrain_models/_emotion_pretrain_all_data_30'  
    if args.model_name == 'roberta':
        config_path = '/home/ythuo/roberta'
    else:
        config_path = '/home/ythuo/bert/'
    args.config =  AutoConfig.from_pretrained(config_path)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)
    # if args.embedding_type == 'roberta':
    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    args.tokenizer = tokenizer
    f_train = open('train_data.tsv','r',encoding='utf-8',newline='')
    csv_train = csv.reader(f_train)
    train_data = []
    train_label = []
    for item in csv_train:
        train_data.append(item[0])
        train_label.append(item[1])

    f_test = open('test_data.tsv','r',encoding='utf-8',newline='')
    csv_test = csv.reader(f_test)
    test_data = []
    test_label = []
    for item in csv_test:
        test_data.append(item[0])
        test_label.append(item[1])
    # print(train_data[1], train_label[1])
    train_dataset = emotion_dataset(args, train_data, train_label)
    test_dataset = emotion_test_dataset(args, test_data, test_label)
    args.path = roberta_path
    pretrain_model = roberta_pretrain(args)
    if args.use_pretrain == 1.:
        pretrain_model.load_state_dict(torch.load(model_path))
    pretrain_model.to(device)
    args.pretrain_model = pretrain_model
    # args.config = 
    # model = AutoModelForMaskedLM.from_pretrained(roberta_path)
    # outputs = model('爱吃西红柿')
    # print(outputs)
    # input()
    model = roberta_classify_test(args)
    model.to(device)
    args.model = model

    train(args, model, train_dataset, test_dataset)
    # results = train(args, model, )
    



if __name__=='__main__':
    main()