import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig, logging
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import AutoTokenizer, AutoModelForMaskedLM,AutoModelForMultipleChoice, AutoModel
# , BertPretrainingHeads
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from transformers.activations import ACT2FN
# from transformers.modeling_outputs import AutoLMHead
# from transformers import BaseAutoModelClass
# from transformers import AutoTokenizer, AutoModelForMaskedLM



class roberta_classify(nn.Module):
    def __init__(self, args):
        super().__init__()
        # print(config.dropout)
        # input()
        self.roberta = AutoModel.from_pretrained(args.path)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.out_proj = nn.Linear(args.hidden_size, args.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=False):
        x,_ = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=return_dict).to_tuple()
        # for i in x:
        #     print(i)
        #     input()
        x = x[:,0,:]
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class roberta_classify_test(nn.Module):
    def __init__(self, args):
        super().__init__()
        # print(config.dropout)
        # input()
        self.roberta = args.pretrain_model.roberta
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.out_proj = nn.Linear(args.hidden_size, args.num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=False):
        x,_ = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=return_dict).to_tuple()
        # for i in x:
        #     print(i)
        #     input()
        x = x[:,0,:]
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class roberta_pretrain(BertPreTrainedModel):
    def __init__(self, args):
        super().__init__(args.config)
        self.roberta = AutoModel.from_pretrained(args.path)
        # self.cls = AutoPre
        self.dense = nn.Linear(args.config.hidden_size, args.config.hidden_size)
        self.transform_act_fn =  ACT2FN[args.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(args.config.hidden_size, eps=args.config.layer_norm_eps)
        self.decoder = nn.Linear(args.config.hidden_size, args.config.vocab_size, bias=False)
        self.decoder.bias = nn.Parameter(torch.zeros(args.config.vocab_size))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, return_dict=False):
        x = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, return_dict=return_dict).to_tuple()
        x = x[0]
        x = self.dense(x)
        x = self.transform_act_fn(x)
        x = self.LayerNorm(x)
        x = self.decoder(x)     # batch_size * vocab_size
        return x


# class roberta(AutoModelForMaskedLM):
#     def __init__(self, config):
#         super.__init__(config)

#     def forw
