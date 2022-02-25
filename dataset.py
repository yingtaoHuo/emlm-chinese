from xml.etree.ElementTree import ProcessingInstruction
from torch.utils.data import Dataset
import torch
import random
import csv
import os
from torch.utils.data import DataLoader
from collate_fn import my_collate_fn, pretrain_collate_fn
from transformers import AutoTokenizer, AutoModelForMaskedLM
from args import parse_args
from ltp import LTP
from cnsenti import Sentiment
import collections
# import tokenization
from bert import tokenization
import numpy as np
from numpy import inf
from emotionDict import emotionDict
from multiprocessing import Pool
import multiprocessing as mp_
import torch.multiprocessing as mp

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def softmax(x):
    x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return x

class emotion_dataset(Dataset):
    def __init__(self,args, data, label):
        self.data = data
        self.args = args
        self.label = label
        self.convert_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        items = self.sentence_ids[idx], \
            self.token_type_ids[idx], \
            self.attention_mask[idx], \
                self.label_ids[idx]
        items_tensor = tuple(torch.tensor(i) for i in items)
        return items_tensor

    def convert_data(self):
        self.sentence_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_ids = []
        emotion_dict = {
            'surprise': 0,
            'sadness':1,
            'anger':2,
            'disgust':3,
            'fear':4,
            'happiness':5,
            'like':6,
            # 'sadness':7,
            'none':7,
        }
        for index in range(len(self.data)):
            dict_ = self.args.tokenizer(self.data[index])
            # print(self.data[index],dict_['input_ids'])
            # input()
            self.sentence_ids.append(dict_['input_ids'])
            self.token_type_ids.append(dict_['token_type_ids'])
            self.attention_mask.append(dict_['attention_mask'])
            self.label_ids.append(emotion_dict[self.label[index]])
        # print(self.sentence_ids)
        # print(self.token_type_ids)
        # input()

class emotion_test_dataset(Dataset):
    def __init__(self,args, data, label):
        self.data = data
        self.args = args
        self.label = label
        self.convert_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        items = self.sentence_ids[idx], \
            self.token_type_ids[idx], \
            self.attention_mask[idx], \
                self.label_ids[idx]
        items_tensor = tuple(torch.tensor(i) for i in items)
        return items_tensor

    def convert_data(self):
        self.sentence_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_ids = []
        emotion_dict = {
            '惊讶': 0,
            '悲伤':1,
            '愤怒':2,
            '厌恶':3,
            '恐惧':4,
            '高兴':5,
            '喜好':6,
            # '悲伤':7,
            '无':7,
        }
        for index in range(len(self.data)):
            # self.data[index] = '[CLS]' + self.data[index] + '[SEP]'
            # print(self.data[index])
            # ef = self.args.tokenizer.tokenize(self.data[index])
            dict_ = self.args.tokenizer(self.data[index])
            # print(ef, dict_)
            # print(len(ef), len(dict_['input_ids']))
            # input()
            self.sentence_ids.append(dict_['input_ids'])
            self.token_type_ids.append(dict_['token_type_ids'])
            self.attention_mask.append(dict_['attention_mask'])
            try:
                self.label_ids.append(emotion_dict[self.label[index]])
            except Exception:
                continue
                print('error:',self.label[index], self.data[index])

class train_instance(object):
    def __init__(self, tokens, masked_lm_positions, masked_lm_labels):
        self.tokens = tokens[0]
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        print(self.tokens)
        input()
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        # s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        # s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        print(self.masked_lm_labels)
        input()
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

class pretrain_dataset(Dataset):
    def __init__(self, args, data):
        self.data = data
        self.args = args
        self.rng = random.Random(args.mask_seed)
        self.emotion_mask()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        items = self.x[idx], \
            self.position[idx], \
            self.label[idx]
        items_tensor = tuple(torch.tensor(i) for i in items)
        return items_tensor

    def emotion_mask(self):
        self.sentence_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_ids = []
        emotion_dict = {
            'surprise': 0,
            'sadness':1,
            'anger':2,
            'disgust':3,
            'fear':4,
            'happiness':5,
            'like':6,
            # '':7,
            'none':7,
            '惊讶': 0,
            '悲伤':1,
            '愤怒':2,
            '厌恶':3,
            '恐惧':4,
            '高兴':5,
            '喜好':6,
            '无':7,
        }
        emotion_mask_sentence = 'emotion_mask.sentence'
        emotion_mask_label = 'emotion_mask.label'
        emotion_mask_position = 'emotion_mask.position'
        if os.path.exists(emotion_mask_sentence):
            self.x = []
            self.position = []
            self.label = []
            file_sentence = open(emotion_mask_sentence, 'r')
            file_label = open(emotion_mask_label, 'r')
            file_position = open(emotion_mask_position, 'r')
            for line in file_sentence.readlines():
                line = line.strip('\n').split(' ')
                for i in range(len(line)):
                    line[i] = int(line[i])
                self.x.append(line)
            for line in file_label.readlines():
                line = line.strip('\n').split(' ')
                for i in range(len(line)):
                    line[i] = int(line[i])
                self.label.append(line)
            for line in file_position.readlines():
                line = line.strip('\n').split(' ')
                for i in range(len(line)):
                    line[i] = int(line[i])
                self.position.append(line)
            file_sentence.close()
            file_label.close()
            file_position.close()
        else:  
            # self.x, self.position, self.label = self.create_training_instance(input_=self.data, tokenizer = self.args.tokenizer, masked_lm_prob=self.args.masked_lm_prob, rng = self.rng)
            # self.x, self.position, self.label = self.create_training_instance(input_=self.data[0:10], tokenizer = self.args.tokenizer, masked_lm_prob=self.args.masked_lm_prob, rng = self.rng)
            # print(self.x)
            # input()
            ctx = mp_.get_context("spawn")
            self.x = []
            self.position = []
            self.label = []
            processor = 10
            process = []
            p = ctx.Pool(processor)
            for i in range(processor):
                process.append(p.apply_async(self.create_training_instance, args=(self.data[i*20000:(i+1)*20000], self.args.tokenizer, self.args.masked_lm_prob, self.rng)))
                # print(self.data[i*10:(i+1)*10])
                # process.append(p.apply_async(self.create_training_instance, args=(self.data[i*10:(i+1)*10], self.args.tokenizer, self.args.masked_lm_prob, self.rng)))
            # for i in process:
            #     i.join()
            #     x, position, label  = i.get()[0], i.get()[1], i.get()[2]
            #     self.x.extend(x)
            #     self.position.extend(position)
            #     self.label.extend(label)
            p.close()
            p.join()
            for i in process:
                x, position, label  = i.get()[0], i.get()[1], i.get()[2]
                self.x.extend(x)
                self.position.extend(position)
                self.label.extend(label)
            # print(self.label, len(self.label))
            fp = open(emotion_mask_sentence, 'w')
            for i in range(len(self.x)):
                fp.write(' '.join(str(j) for j in self.x[i]))
                fp.write('\n')
            fp.close()
            fp = open(emotion_mask_label, 'w')
            for i in range(len(self.label)):
                fp.write(' '.join(str(j) for j in self.label[i]))
                fp.write('\n')
            fp.close()
            fp = open(emotion_mask_position, 'w')
            for i in range(len(self.position)):
                fp.write(' '.join(str(j) for j in self.position[i]))
                fp.write('\n')
            fp.close()
        # input()

    def create_training_instance(self, input_, tokenizer, masked_lm_prob,rng,min_length=5,max_length=12):
        all_wholeword = []
        all_pieceword = []
        ltp = LTP()
        num = 0
        # print(input_)
        for i in input_:
            a_ = []
            a_.append(i)
            try:
                tokens = ltp.seg(a_)
            except:
                continue
                print('error:',a_, num)
            num += 1
            a_ = tokens[0]
            # print(a_)
            if len(a_[0])  <= min_length or len(a_[0]) >= max_length:
                continue
            all_wholeword.append(a_[0])
            tokens = tokenizer.tokenize(i)
            all_pieceword.append(tokens)
        instances = []
        vocab_words = list(tokenizer.vocab.keys())
        # print(all_pieceword)
        for i in range(len(all_pieceword)):
            if len(all_wholeword[i]) <= min_length:
                continue
            instances.append(create_instances_from_document(all_pieceword[i], all_wholeword[i], vocab_words=vocab_words, rng = rng))
        # return all_pieceword, 
        x_ = []
        position_ = []
        label_ = []
        for i in range(len(all_wholeword)):
            # print(all_wholeword[i])
            x_.append(self.args.tokenizer.convert_tokens_to_ids(instances[i][0]))
            position_.append(instances[i][1])
            # label_now = []
            # label_.append(list(map(int,self.args.tokenizer.convert_tokens_to_ids(instances[i][2]))))
            # for ex in self.args.tokenizer.convert_tokens_to_ids(instances[i][2]):
            #     label_now.append(long(ex))
            # # print(self.args.tokenizer.convert_tokens_to_ids(instances[i][2]))
            # # input()
            label_.append(self.args.tokenizer.convert_tokens_to_ids(instances[i][2]))
        #return torch.tensor(x_), torch.tensor(position_), torch.tensor(label_)
        # print(label_)
        # input()
        return x_, position_, label_

def create_instances_from_document(all_pieceword, all_wholeword,  masked_lm_prob=0.15,max_predictons_per_seq=3, vocab_words=None, rng=None):
    instances = []
    emotion_dict = emotionDict()
    all_pieceword.insert(0,'[CLS]')
    all_pieceword.append('[SEP]')
    # print(all_pieceword)
    segment_ids = [0 for _ in range(len(all_pieceword))]
    all_wholeword.insert(0,'[CLS]')
    all_wholeword.append('[SEP]')
    for i in all_pieceword:
        instances.append(i)
    # print(all_wholeword, cand_indexes)
    # input()
    num_to_predict = min(max_predictons_per_seq, max(0, int(round(len(all_wholeword) * masked_lm_prob))))
    masked_lms = []
    pre_len = 1
    # print(all_wholeword)
    # print(all_pieceword)
    mask_prob = np.zeros((1,len(all_wholeword)))
    prob_all = 0
    non_emotion_num = 0
    for (_, word) in enumerate(all_wholeword):
        word_ = emotion_dict.evaluate(word)
        if word_:
            # print(word_)
            # input()
            mask_prob[0][_] = min(word_[2]/6,1.)
            prob_all += mask_prob[0][_]
        else:
            # print(_)
            mask_prob[0][_] = -inf
            non_emotion_num += 1
    # print(non_emotion_num, len(all_wholeword), prob_all)
    for i in range(len(mask_prob[0])):
        score= max(len(all_wholeword)*0.15-prob_all-0.3,0)/(non_emotion_num-2)
        if mask_prob[0][i] == -inf:
            mask_prob[0][i] = score
    # print(mask_prob)
    # print(all_wholeword)
    # input()
    # mask_prob = softmax(mask_prob)
    # print(mask_prob)
    # input()
    b = mask_prob.shape[1]
    # for i in range(1,b):
    #     mask_prob[0][i] += mask_prob[0][i-1]
    # print(mask_prob)
    # input()
    num = 0
    while len(masked_lms) < num_to_predict:
        # num += 1
        # if num > 10:
        #     break
        pre_len = 1
        for (_,word) in enumerate(all_wholeword):
            if word == '[SEP]' or word == '[CLS]':
                continue
            if len(masked_lms) >= num_to_predict:
                break
            if word[0] >= '0' and word[0] <= '9':
                pre_len += 1
                continue
            # print(rng.random())
            a = rng.random()
            if a > mask_prob[0][_]:
                pre_len += len(word)
                continue
            # if rng.random() > 0.3:
            #     pre_len += len(word)
            #     continue
            if rng.random() < 0.8:
                masked_token = "[MASK]"
                # print(word, len(word), _, all_wholeword[_])
                # input()
                try:
                    for i in range(len(word)):
                        all_pieceword[i+pre_len] = masked_token
                except:
                    continue
            else:
                if rng.random() < 0.5:
                    pre_len += len(word)
                    continue
                else:
                    try:
                        for i in range(len(word)):
                            all_pieceword[i+pre_len] = vocab_words[rng.randint(0,len(vocab_words) - 1)]
                    except:
                        continue
                        # print(word)
                        # print(all_pieceword)
                        # input()
            for i in range(len(word)):
                try:
                    masked_lms.append(MaskedLmInstance(index=pre_len+i, label=instances[pre_len+i]))
                except:
                    continue
            pre_len += len(word)
    masked_lms = sorted(masked_lms, key=lambda x:x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (all_pieceword, masked_lm_positions, masked_lm_labels)

if __name__=='__main__':
    roberta_path = '/home/ythuo/roberta'
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    f_train = open('pretrain_data_tsv_split','r',encoding='utf-8',newline='')
    # f_train = open('train_data.tsv','r',encoding='utf-8',newline='')
    csv_train = csv.reader(f_train)
    train_data = []
    train_label = []
    for item in csv_train:
        train_data.append(item[0])
        train_label.append(item[1])
    args.tokenizer = AutoTokenizer.from_pretrained(roberta_path)
    pretrain_dataset = pretrain_dataset(args,data=train_data)
    # pretrain_dataset = emotion_dataset(args, data=train_data,label=train_label)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.train_batch_size,\
        collate_fn = pretrain_collate_fn, shuffle=True)