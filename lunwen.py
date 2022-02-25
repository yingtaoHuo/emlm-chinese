from random import Random
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import RobertaModel, AutoConfig
import torch.nn.functional as F
import pretrain
import csv
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
from transformers import AdamW
roberta_path = '/home/ythuo/roberta'
from dataset import emotion_dataset, emotion_test_dataset, pretrain_dataset
from collate_fn import my_collate_fn, pretrain_collate_fn
from args import parse_args
from model import roberta_classify, roberta_pretrain
import os
from torch.nn import CrossEntropyLoss
# from transformers import RobertaTokenizerFast
# roberta_path = '/home/ythuo/roberta'
# tokenizer = RobertaTokenizerFast.from_pretrained(roberta_path)
logger = logging.getLogger(__name__)

def save_model(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Seving model checkpoint to {output_dir}")
    #if not isinstance(self.model, PretrainedModel)


def get_roberta_optimizer(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
            'weight_decay':args.weight_decay},
        {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            'weight_decay': 0}
    ]
    optimizer = AdamW(optimizer_parameters, lr=args.learning_rate,\
        eps=args.adam_epsilon)

    return optimizer

def pretrain(args, pretrain_data):
    path = '/home/ythuo/weibo/pretrain_models/'
    model = args.model
    pretrain_loss = 0
    optimizer = get_roberta_optimizer(args, model)
    all_eval_results = []
    # num = 0
    pretrain_loss = 0
    train_iterator = trange(int(args.epochs), desc='Epoch')
    #　num_update_steps_epoch = math.ceil(args.)
    pretrain_num = 0
    for _ in train_iterator:
        pretraindataset = pretrain_dataset(args, data=pretrain_data)
        pretrain_dataloader = DataLoader(pretraindataset, batch_size=args.pretrain_batch_size,\
        collate_fn = pretrain_collate_fn, shuffle=True)
        model.train()
        model.zero_grad()
        pretrain_num += 1
        for batch in pretrain_dataloader:
            input_ids = batch[0].to(args.device)
            label_position_ids = batch[1]
            label_ids = batch[2]
            output = model(input_ids=input_ids, return_dict=True)
            a, b, c = output.size()
            select_representation = None
            labels = None
            for i in range(a):
                output_select = torch.zeros((len(label_position_ids[i]), b)).to(args.device)
                for j in range(len(label_position_ids[i])):
                    output_select[j][label_position_ids[i][j]] = 1
                if select_representation is None:
                    select_representation = torch.mm(output_select, output[i])
                else:
                    select_representation = torch.cat((select_representation, torch.mm(output_select, output[i])))
                if labels is None:
                    labels = torch.tensor(label_ids[i])
                else:
                    labels = torch.cat((labels, torch.tensor(label_ids[i])))
            labels = labels.to(args.device)
            labels = labels.long()
            loss = F.cross_entropy(select_representation, labels)
            loss.backward()
            loss_ = loss.item()
            optimizer.step()
            model.zero_grad()
            torch.cuda.empty_cache()
            pretrain_loss += loss_
    # path = ''.join(path+'_emotion_pretrain_all_data_'+str(pretrain_num))
    # torch.save(model.state_dict(),path)


def main():
    args = parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)
    tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    args.tokenizer = tokenizer
    # f_train = open('pretrain_data_tsv_split','r',encoding='utf-8',newline='')
    f_train = open('pretrain_data_tsv_split','r',encoding='utf-8',newline='')
    csv_train = csv.reader(f_train)
    train_data = []
    train_label = []
    for item in csv_train:
        if len(item) == 0:
            continue
        train_data.append(item[0])
        # train_label.append(item[1])
    # print(train_data[1], train_label[1])
    # train_dataset = pretrain_dataset(args, data=train_data)
    args.path = roberta_path
    config = AutoConfig.from_pretrained(args.path)
    args.config = config
    model = roberta_pretrain(args)
    model.to(device)
    args.model = model
    pretrain(args, train_data)
    



if __name__=='__main__':
    main()