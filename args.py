import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='roberta', choices=['roberta','bert','bert-wwm'])
    #parser.add_argument('--output_dir', type=str, default='/')
    parser.add_argument('--bert_model_dir', type=str,default='/home/ythuo/bert/',help= 'raw_bert_model')
    parser.add_argument('--seed', type=int, default=2021,
                        help='random seed for initialization')
    parser.add_argument('--mask_seed', type=int, default=2021,
                        help='random seed for initialization')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--pretrain_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--cuda_id', type=str, default='0')
    parser.add_argument('--num_labels', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--pretrain_learning_rate', type=float, default=5e-6)
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--max_grad_norm',default=1.0,type=float)
    parser.add_argument('--use_pretrain',default=1.,type=float)
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for embedding.')
    parser.add_argument('--hidden_size', type=int, default=768)

    

    '''
    pingjie mode:
    0:  no change
    1:  add doubt
    2:  add state
    3:  add doubt and state
    '''
    # parser.add_argument('--')
    # parser.add_argument('--device', type=)
    return parser.parse_args()

if __name__=='__main__':
    print('args')