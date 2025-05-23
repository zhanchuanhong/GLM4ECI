import torch


def getEncTemplate(sentence1_id,sentence2_id,e1_id,e2_id,s_1,s_2):
    if sentence1_id == sentence2_id:
        clabel_1 = 0                    # 句内句间标志,0表示句内，1表示句间
        if int(e1_id[1]) > int(e2_id[1]):
            s_1.insert(int(e1_id[1]), '<c1>')
            s_1.insert(int(e1_id[1]) + len(e1_id), '</c1>')
            s_1.insert(int(e2_id[1]), '<c2>')
            s_1.insert(int(e2_id[1]) + len(e2_id), '</c2>')
        else:
            s_1.insert(int(e2_id[1]), '<c2>')
            s_1.insert(int(e2_id[1]) + len(e2_id), '</c2>')
            s_1.insert(int(e1_id[1]), '<c1>')
            s_1.insert(int(e1_id[1]) + len(e1_id), '</c1>')
        template = " ".join(s_1)
    else:
        clabel_1=1                      # 句内句间标志,0表示句内，1表示句间
        s_1.insert(int(e1_id[1]), '<c1>')
        s_1.insert(int(e1_id[1]) + len(e1_id), '</c1>')
        s_2.insert(int(e2_id[1]), '<c2>')
        s_2.insert(int(e2_id[1]) + len(e2_id), '</c2>')
        s_1 = " ".join(s_1)
        s_2 = " ".join(s_2)
        if sentence1_id < sentence2_id:
            template = s_1 + ' <s> ' + s_2
        else:
            template = s_2 + ' </s> ' + s_1
    return template,clabel_1


def getDecTemplate(data,flag):
    dec_template = 'In this sentence , <mask>causes <mask2> .'
    return dec_template


# tokenize sentence and get event idx
def get_batch(data, args, indices, tokenizer, flag):
    batch_enc_idx, batch_enc_mask, batch_dec_idx, batch_dec_mask, casual_label, label_b, label_c, clabel_b = [], [], [], [], [], [], [], []
    for idx in indices:
        label, label_1, label_2, clabel_1 = 1, 1, 1, 1
        e1_id, e2_id, sentence1_id, sentence2_id, s_1, s_2 = data[idx][14], data[idx][15], data[idx][11], data[idx][13], data[idx][10], data[idx][12]
        s_1 = s_1.split()[0:int((args.len_enc_arg)/2)]
        s_2 = s_2.split()[0:int((args.len_enc_arg)/2)]
        e1_id = e1_id.split("_")
        e2_id = e2_id.split("_")

        enc_template, clabel_1 = getEncTemplate(sentence1_id,sentence2_id,e1_id,e2_id,s_1,s_2)
        encoder_dict = tokenizer.encode_plus(
            enc_template,
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_enc_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        batch_enc_idx.append(encoder_dict['input_ids'])
        batch_enc_mask.append(encoder_dict['attention_mask'])

        dec_template = getDecTemplate(data[idx],flag)
        decoder_dict = tokenizer.encode_plus(
            dec_template,
            add_special_tokens=True,
            padding='max_length',
            max_length=args.len_dec_arg,
            truncation=True,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        batch_dec_idx.append(decoder_dict['input_ids'])
        batch_dec_mask.append(decoder_dict['attention_mask'])
        assert decoder_dict['input_ids'][0][5].item() in [50264,50265,50266,50267,50270]
        assert decoder_dict['input_ids'][0][8].item() in [50264,50265,50266,50267,50270]

        if data[idx][9] == 'NONE':
            label = 0
            label_1 = tokenizer('<na>')['input_ids'][1:-1]  # 50270
            label_2 = tokenizer('<na>')['input_ids'][1:-1]  # 50270
        elif data[idx][9] == 'PRECONDITION':
            label = 1
            label_1 = tokenizer('<c1>')['input_ids'][1:-1]  # 50266
            label_2 = tokenizer('<c2>')['input_ids'][1:-1]  # 50267
        elif data[idx][9] == 'FALLING_ACTION':
            label = 2
            label_1 = tokenizer('<c2>')['input_ids'][1:-1]  # 50267
            label_2 = tokenizer('<c1>')['input_ids'][1:-1]  # 50266
        label_b += label_1
        label_c += label_2
        casual_label.append(label)
        clabel_b.append(clabel_1)   # 句内句间标志,0表示句内，1表示句间

    batch_enc_idx = torch.cat(batch_enc_idx, dim=0)
    batch_enc_mask = torch.cat(batch_enc_mask, dim=0)
    batch_dec_idx = torch.cat(batch_dec_idx, dim=0)
    batch_dec_idx[:, 0] = tokenizer.eos_token_id
    batch_dec_mask = torch.cat(batch_dec_mask, dim=0)

    return batch_enc_idx, batch_enc_mask, batch_dec_idx, batch_dec_mask, casual_label, label_b, label_c, clabel_b


def cal_handler(label, predt):
    confusion_matrix = [[0, 0],
                        [0, 0]]
    for i in range(len(label)):
        confusion_matrix[predt[i]][label[i]] += 1
    Precision = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1]+ 1e-9)
    Recall = confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][1] + 1e-9)
    F1 = 2 * Precision * Recall / (Precision + Recall + 1e-9)
    return Precision, Recall, F1, confusion_matrix


def calculate(label_direction, pred_direction, clabel):
    intra_label,intra_predt,cross_label,cross_predt = [],[],[],[]
    assert len(label_direction) == len(pred_direction)
    assert len(label_direction) == len(clabel)
    for i in range(len(clabel)):
        if clabel[i] == 0:
            intra_label.append(label_direction[i])
            intra_predt.append(pred_direction[i])
        else:
            cross_label.append(label_direction[i])
            cross_predt.append(pred_direction[i])
    intra_p,intra_r,intra_f,_ = cal_handler(intra_label, intra_predt)
    cross_p,cross_r,cross_f,_ = cal_handler(cross_label, cross_predt)
    all_p,all_r,all_f,confusion_matrix = cal_handler(label_direction, pred_direction)
    return intra_p,intra_r,intra_f, cross_p,cross_r,cross_f, all_p,all_r,all_f, confusion_matrix


def cal_dir_handler(label, predt):
    confusion_matrix = [[0,0,0],
                        [0,0,0],
                        [0,0,0]]
    for i in range(len(label)):
        confusion_matrix[predt[i]][label[i]] += 1
    Precision = (confusion_matrix[1][1]+confusion_matrix[2][2])/(confusion_matrix[1][0]+confusion_matrix[1][1]+confusion_matrix[1][2]+
                                                                 confusion_matrix[2][0]+confusion_matrix[2][1]+confusion_matrix[2][2]+1e-9)
    Recall = (confusion_matrix[1][1]+confusion_matrix[2][2])/(confusion_matrix[0][1]+confusion_matrix[0][2]+
                                                              confusion_matrix[1][1]+confusion_matrix[1][2]+
                                                              confusion_matrix[2][1]+confusion_matrix[2][2]+1e-9)
    F1 = 2*Precision*Recall/(Precision+Recall+1e-9)
    return Precision,Recall,F1,confusion_matrix

def calculate_direction(label_direction, pred_direction, clabel):
    intra_label,intra_predt,cross_label,cross_predt = [],[],[],[]
    assert len(label_direction) == len(pred_direction)
    assert len(label_direction) == len(clabel)
    for i in range(len(clabel)):
        if clabel[i] == 0:
            intra_label.append(label_direction[i])
            intra_predt.append(pred_direction[i])
        else:
            cross_label.append(label_direction[i])
            cross_predt.append(pred_direction[i])
    intra_p,intra_r,intra_f,_ = cal_dir_handler(intra_label, intra_predt)
    cross_p,cross_r,cross_f,_ = cal_dir_handler(cross_label, cross_predt)
    all_p,all_r,all_f,confusion_matrix = cal_dir_handler(label_direction, pred_direction)
    return intra_p,intra_r,intra_f, cross_p,cross_r,cross_f, all_p,all_r,all_f, confusion_matrix
