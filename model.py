# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
embedding_size = 768


class MLP(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bart_model = BartForConditionalGeneration.from_pretrained(args.model_name).to(device)
        self.bart_model.resize_token_embeddings(args.vocab_size)
        for param in self.bart_model.parameters():
            param.requires_grad = True

        self.hidden_size = 768

        self.vocab_size = args.vocab_size

    def forward(self, batch_enc_idx, batch_enc_mask, batch_dec_idx, batch_dec_mask, flag):
        answer_space = [50266,50267,50270]
        if flag == 'train':
            outputs = self.bart_model(input_ids=batch_enc_idx,
                                      attention_mask=batch_enc_mask,
                                      decoder_input_ids=batch_dec_idx,
                                      decoder_attention_mask=batch_dec_mask,
                                      return_dict=True)
            prediction1 = outputs['logits'][:, 4, :]
            prediction2 = outputs['logits'][:, 7, :]
            return prediction1, prediction2
        else:
            outputs = self.bart_model(input_ids=batch_enc_idx,
                                      attention_mask=batch_enc_mask,
                                      decoder_input_ids=batch_dec_idx,
                                      decoder_attention_mask=batch_dec_mask,
                                      return_dict=True)
            prediction1 = outputs['logits'][:, 4, answer_space]
            prediction2 = outputs['logits'][:, 7, answer_space]
            return prediction1, prediction2

    def handler(self, to_add, tokenizer):
        da = self.bart_model.model.shared.weight
        # da = self.bart_model.roberta.embeddings.word_embeddings.weight
        for i in to_add.keys():
            l = to_add[i]
            with torch.no_grad():
                temp = torch.zeros(self.hidden_size).to(device)
                for j in l:
                    temp += da[j]
                temp /= len(l)

                da[tokenizer.convert_tokens_to_ids(i)] = temp
