import torch
def pad_sequence(sequence, batch_first=False, padding_value=0.0):
    max_len = max([i.size(0) for i in sequence])
    max_size = sequence[0].size()
    trailing_dims = max_size[1:]
    if batch_first:
        out_dims = (len(sequence), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequence)) + trailing_dims
    #print(sequence[0].size())
    out_tensor = sequence[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequence):
        length = sequence[i].size(0)
        if batch_first == True:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length,i,...] = tensor
    return out_tensor


def my_collate_fn(batch):
    sentence_ids, token_type_ids, attention_ids, labels = zip(*batch)

    sentence_ids = pad_sequence(sentence_ids, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attention_ids = pad_sequence(attention_ids, batch_first=True, padding_value=0)
    # print(sentence_ids)
    # input()
    return sentence_ids, attention_ids, token_type_ids, labels 

def pretrain_collate_fn(batch):
    sentence_ids, label_position_ids, label_ids = zip(*batch)
    sentence_ids = pad_sequence(sentence_ids, batch_first=True, padding_value=0)
    # print(sentence_ids)
    # input()
    # label_position_ids = pad_sequence(label_position_ids, batch_first=True, padding_value=0)
    # label_ids = pad_sequence(label_ids, batch_first=True, padding_value=0)
    return sentence_ids, label_position_ids, label_ids 