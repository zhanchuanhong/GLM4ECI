import torch

# In ESC data sets, some event-mentions are not continuous.
# put ...  on ---> put Tompsion on
def isContinue(id_list):
    for i in range(len(id_list)-1):
        if int(id_list[i])!=int(id_list[i+1])-1:
            return False
    return True

def correct_data(data):
    for i in range(len(data)):
        e1_id = data[i][14].split('_')[1:]
        e2_id = data[i][15].split('_')[1:]
        if not isContinue(e1_id):
            s_1 = data[i][10].split()
            event1=s_1[int(e1_id[0]):int(e1_id[-1])+1]
            event1=' '.join(event1)
            event1+=' '
            new_e1_id=[str(i) for i in range(int(e1_id[0]),int(e1_id[-1])+1)]
            event_place1='_'+'_'.join(new_e1_id)
            sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], event1, data[i][8],data[i][9], data[i][10],
                        data[i][11], data[i][12], data[i][13], event_place1, data[i][15])
            data.pop(i)
            data.insert(i, sentence)
        if not isContinue(e2_id):
            s_2 = data[i][12].split()
            event2=s_2[int(e2_id[0]):int(e2_id[-1])+1]
            event2=' '.join(event2)
            event2+=' '
            new_e2_id=[str(i) for i in range(int(e2_id[0]),int(e2_id[-1])+1)]
            event_place2='_'+'_'.join(new_e2_id)
            sentence=(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],data[i][7],event2,data[i][9],data[i][10],
                      data[i][11],data[i][12],data[i][13],data[i][14],event_place2)
            data.pop(i)
            data.insert(i, sentence)
    return data


# collect multi-token event-mentions
def collect_mult_event(train_data,tokenizer):
    multi_event=[]
    to_add={}
    special_multi_event_token=[]
    event_dict={}
    reverse_event_dict={}
    for sentence in train_data:
        if len(tokenizer(' '+sentence[7].strip())['input_ids'][1:-1])>1 and sentence[7] not in multi_event:
            multi_event.append(sentence[7])
            special_multi_event_token.append("<a_"+str(len(special_multi_event_token))+">")
            event_dict[special_multi_event_token[-1]]=multi_event[-1]
            reverse_event_dict[multi_event[-1]]=special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]]=tokenizer(multi_event[-1].strip())['input_ids'][1: -1]
        if len(tokenizer(' '+sentence[8].strip())['input_ids'][1:-1])>1 and sentence[8] not in multi_event:
            multi_event.append(sentence[8])
            special_multi_event_token.append("<a_"+str(len(special_multi_event_token))+">")
            event_dict[special_multi_event_token[-1]]=multi_event[-1]
            reverse_event_dict[multi_event[-1]]=special_multi_event_token[-1]
            to_add[special_multi_event_token[-1]] = tokenizer(multi_event[-1].strip())['input_ids'][1: -1]

    return multi_event,special_multi_event_token,event_dict,reverse_event_dict,to_add


# Replace multi-token events with special characters <A-i>
# For example：He has went to the school.--->He <A_3> the school.
def replace_mult_event(data,reverse_event_dict):
    for i in range(len(data)):
        if (data[i][7] in reverse_event_dict) and (data[i][8] not in reverse_event_dict):
            s_1 = data[i][10].split()
            e1_id = data[i][14].split('_')[1:]
            e1_id.reverse()
            for id in e1_id:
                s_1.pop(int(id))
            s_1.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])
            sentence=(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4],data[i][5],data[i][6],reverse_event_dict[data[i][7]],data[i][8],data[i][9]," ".join(s_1),
                          data[i][11],data[i][12],data[i][13],'_'+e1_id[-1],data[i][15])
            data.pop(i)
            data.insert(i,sentence)
        if (data[i][7] not in reverse_event_dict) and (data[i][8] in reverse_event_dict):
            s_2 = data[i][12].split()
            e2_id = data[i][15].split('_')[1:]
            e2_id.reverse()
            for id in e2_id:
                s_2.pop(int(id))
            s_2.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
            sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7], reverse_event_dict[data[i][8]],data[i][9], data[i][10],
                            data[i][11], " ".join(s_2), data[i][13], data[i][14], '_'+e2_id[-1])
            data.pop(i)
            data.insert(i, sentence)
        if (data[i][7] in reverse_event_dict) and (data[i][8] in reverse_event_dict):
            e1_id = data[i][14].split('_')[1:]
            e2_id = data[i][15].split('_')[1:]
            e1_id.reverse()
            e2_id.reverse()
            if data[i][11] == data[i][13]:
                s = data[i][10].split()
                if int(e1_id[0])<int(e2_id[0]):
                    for id in e2_id:
                        s.pop(int(id))
                    s.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
                    for id in e1_id:
                        s.pop(int(id))
                    s.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])
                else:
                    for id in e1_id:
                        s.pop(int(id))
                    s.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])
                    for id in e2_id:
                        s.pop(int(id))
                    s.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
                sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], reverse_event_dict[data[i][7]],reverse_event_dict[data[i][8]],data[i][9], " ".join(s),
                            data[i][11], " ".join(s), data[i][13], '_'+str(s.index(reverse_event_dict[data[i][7]])), '_'+str(s.index(reverse_event_dict[data[i][8]])))
                data.pop(i)
                data.insert(i, sentence)
            if data[i][11] != data[i][13]:
                s_1 = data[i][10].split()
                for id in e1_id:
                    s_1.pop(int(id))
                s_1.insert(int(e1_id[-1]), reverse_event_dict[data[i][7]])

                s_2 = data[i][12].split()
                for id in e2_id:
                    s_2.pop(int(id))
                s_2.insert(int(e2_id[-1]), reverse_event_dict[data[i][8]])
                sentence = (data[i][0], data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], reverse_event_dict[data[i][7]],reverse_event_dict[data[i][8]], data[i][9], " ".join(s_1),
                            data[i][11], " ".join(s_2), data[i][13], '_'+str(s_1.index(reverse_event_dict[data[i][7]])), '_'+str(s_2.index(reverse_event_dict[data[i][8]])))
                data.pop(i)
                data.insert(i, sentence)
    return data

def isSort(prediction,label,sort_rate):
    prediction=torch.softmax(prediction,0)
    prediction=prediction.detach().cpu().tolist()
    temp=prediction[label]
    if temp > sort_rate:
        return True
    return False


def convert(prediction,batch_event):
    to_save_prediction = torch.softmax(prediction, dim=1)
    to_save_prediction = to_save_prediction.detach().cpu().tolist()
    event_predt = []
    event_place = []
    nothing_predt = []
    nothing_place = []
    for temp_i in range(len(prediction)):
        event_predt.append(to_save_prediction[temp_i][batch_event])
        nothing_predt.append(to_save_prediction[temp_i][50270])
        to_save_prediction[temp_i].sort(reverse=True)
        to_save_prediction[temp_i] = to_save_prediction[temp_i][:10]

    for temp_i in range(len(prediction)):
        if event_predt[temp_i] in to_save_prediction[temp_i]:
            event_place.append(to_save_prediction[temp_i].index(event_predt[temp_i]))
        else:
            event_place.append(-1)
        if nothing_predt[temp_i] in to_save_prediction[temp_i]:
            nothing_place.append(to_save_prediction[temp_i].index(nothing_predt[temp_i]))
        else:
            nothing_place.append(-1)
    return to_save_prediction,event_place,nothing_place

def getPredt(args,prediction1,prediction2):
    predt_direction = []
    answer_space = [50266, 50267, 50270]    # c1 c2 na
    anser_predt1, anser_predt2 = torch.softmax(prediction1[:, answer_space], dim=1), torch.softmax(prediction2[:, answer_space], dim=1)
    for iii in range(len(prediction1)):  # 当[MASK1]和[MASK2]在NA上的概率和小于rate_sort时，认为有因果
        if anser_predt1[iii][2].item() + anser_predt2[iii][2].item() < args.rate_sort:      # na概率和小于阈值，认为无因果
            if anser_predt1[iii][0] + anser_predt2[iii][1] > anser_predt1[iii][1] + anser_predt2[iii][0]:
                predt_direction.append(1)  # e1 cause e2
            elif anser_predt1[iii][1] + anser_predt2[iii][0] > anser_predt1[iii][0] + anser_predt2[iii][1]:
                predt_direction.append(2)  # e2 cause e1
        else:
            predt_direction.append(0)
    return predt_direction

def assert_handler(all_p,all_r,all_f,matrix,matrix_dir):
    new_mat = [[matrix_dir[0][0],                  matrix_dir[0][1]+matrix_dir[0][2]],
               [matrix_dir[1][0]+matrix_dir[2][0], matrix_dir[1][1]+matrix_dir[1][2]+matrix_dir[2][1]+matrix_dir[2][2]]]
    assert matrix == new_mat
    Precision = new_mat[1][1] / (new_mat[1][0] + new_mat[1][1] + 1e-9)
    Recall = new_mat[1][1] / (new_mat[0][1] + new_mat[1][1] + 1e-9)
    F1 = 2 * Precision * Recall / (Precision + Recall + 1e-9)
    assert Precision == all_p
    assert Recall == all_r
    assert F1 == all_f
    return