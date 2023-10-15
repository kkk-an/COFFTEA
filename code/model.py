from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertConfig
import allennlp.modules.span_extractors.max_pooling_span_extractor as max_pooling_span_extractor
from allennlp.nn.util import get_mask_from_sequence_lengths, masked_log_softmax
import dgl
import dgl.function as fn
import torch


class BertForFrameId(BertPreTrainedModel):
    def __init__(self, config, frame_dropout_prob=None):
        super().__init__(config)
        self.config = config
        # self.sentence_encoder = BertModel(config, add_pooling_layer=False)
        self.sentence_encoder = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
        self.target_extractor = max_pooling_span_extractor.MaxPoolingSpanExtractor(config.hidden_size)
        self.target_dropout = nn.Dropout(frame_dropout_prob if frame_dropout_prob is not None else config.hidden_dropout_prob)
        self.target_classifier = nn.Linear(config.hidden_size, config.frame_numbers)
        
        # self.frame_encoder = BertModel(config, add_pooling_layer=False) 
        self.frame_encoder = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False)
        self.frame_dropout = nn.Dropout(frame_dropout_prob if frame_dropout_prob is not None else config.hidden_dropout_prob)


    def forward(
        self,
        sentence_input_ids=None,
        sentence_attention_mask=None,
        sentence_token_type_ids=None,
        sentence_position_ids=None,
        sentence_head_mask=None,
        target_start_pos=None,
        target_end_pos=None,
        labels=None,           # Determine if loss needs to be calculated
        frame_input_ids=None,
        frame_attention_mask=None,
        frame_token_type_ids=None,
        frame_position_ids=None,
        frame_head_mask=None,
        n_choices=None,         # Used to determine lexical filter
    ):

        if n_choices is not  None: # map each sentence to corresponding frame def
            def remove_zero_vectors(t, n_choices):
                result = 'None'
                for batch, n in zip(t, n_choices):
                    if result == 'None':
                        result = batch[:n]
                    else:
                        result = torch.cat((result, batch[:n]), 0)
                return result

            # inut_ids : bsz x max_choice x max_length
            max_choice = sentence_input_ids.shape[1]
            n_batch = sentence_input_ids.shape[0]

            #  two settings: one is to use Zero padding, and the other is to use negative samples to fill to max_choice
            sentence_input_ids = remove_zero_vectors(sentence_input_ids, n_choices)
            sentence_attention_mask = remove_zero_vectors(sentence_attention_mask, n_choices) if sentence_attention_mask is not None else None
            sentence_token_type_ids = remove_zero_vectors(sentence_token_type_ids, n_choices) if sentence_token_type_ids is not None else None
            sentence_position_ids = remove_zero_vectors(sentence_position_ids, n_choices) if sentence_position_ids is not None else None
            target_start_pos = remove_zero_vectors(target_start_pos, n_choices) if target_start_pos is not None else None
            target_end_pos = remove_zero_vectors(target_end_pos, n_choices) if target_end_pos is not None else None
            frame_input_ids = remove_zero_vectors(frame_input_ids, n_choices)
            frame_attention_mask = remove_zero_vectors(frame_attention_mask, n_choices) if frame_attention_mask is not None else None
            frame_token_type_ids = remove_zero_vectors(frame_token_type_ids, n_choices) if frame_token_type_ids is not None else None
            frame_position_ids = remove_zero_vectors(frame_position_ids, n_choices) if frame_position_ids is not None else None

            sentence_output = self.sentence_encoder(
                input_ids=sentence_input_ids,
                attention_mask=sentence_attention_mask,
                token_type_ids=sentence_token_type_ids,
                position_ids=sentence_position_ids,
                head_mask=sentence_head_mask,
            )

            assert target_start_pos.shape == target_end_pos.shape

            # sentence_output:  sequence_output pooled_output (hidden_states)
            sentence_embed = sentence_output["last_hidden_state"]
            target_start_pos.unsqueeze_(1)
            target_end_pos.unsqueeze_(1)
            target_ids = torch.concat((target_start_pos, target_end_pos), dim=1).unsqueeze(1)
            target_embed = self.target_extractor(sentence_embed, target_ids).squeeze(1) 
            target_embed = self.target_dropout(target_embed)
            # print("target_embed", target_embed.shape)     # bsz x hidden_size

            frame_output = self.frame_encoder(
                input_ids=frame_input_ids,
                attention_mask=frame_attention_mask,
                token_type_ids=frame_token_type_ids,
                position_ids=frame_position_ids,
                head_mask=frame_head_mask,
            )
            # frame_output: mean pooling or [CLS]
            framedef_embed = frame_output["last_hidden_state"][:,1:,:]
            frame_mask = frame_attention_mask[:,1:]     # bsz x seq_len-1
            framedef_embed = framedef_embed.masked_fill(frame_mask.unsqueeze(-1).eq(0), 0).sum(dim=1)
            framedef_embed = framedef_embed / frame_mask.sum(dim=1, keepdim=True).float()
            framedef_embed = self.frame_dropout(framedef_embed)     # bsz x hidden_size
            # print("framedef_embed", framedef_embed.shape)     # bsz x hidden_size

            assert target_embed.shape == framedef_embed.shape, "target's shape is not equal to frame definition's shape"

            similarity = torch.div(F.cosine_similarity(target_embed, framedef_embed, dim=1), 0.07)
            similarity = similarity.view(-1)

            flat_choices = [0] * (len(n_choices) + 1)
            for i in range(1, len(n_choices) + 1):
                flat_choices[i] = flat_choices[i-1] + n_choices[i-1]
            assert flat_choices[-1] == similarity.shape[0]
            """reshaped similarity padding shoule be -inf"""
            reshaped_similarity = torch.zeros([n_batch, max_choice], dtype=similarity.dtype, device=similarity.device) - float("Inf")
            for i in range(n_batch):
                reshaped_similarity[i][:n_choices[i]] = similarity[flat_choices[i]:flat_choices[i+1]]
            # print("reshaped_similarity",reshaped_similarity,reshaped_similarity.shape)

            outputs = (reshaped_similarity, target_embed, framedef_embed)
            if labels is not None:
                loss  = sum(F.cross_entropy(similarity[flat_choices[i]:flat_choices[i+1]].unsqueeze(0), labels[i].unsqueeze(0)) 
                        for i in range(n_batch)) / n_batch
                print(f"      ce_loss: {loss:.5f}")
                outputs = (loss, ) + outputs
            else:
                outputs = (torch.tensor(0, dtype=torch.float), ) + outputs
            return outputs
        
        else:
            # without lf, directly compute CE loss
            sentence_output = self.sentence_encoder(
                input_ids=sentence_input_ids,
                attention_mask=sentence_attention_mask,
                token_type_ids=sentence_token_type_ids,
                position_ids=sentence_position_ids,
                head_mask=sentence_head_mask,
            )

            assert target_start_pos.shape == target_end_pos.shape

            # sentence_output:  sequence_output pooled_output (hidden_states)
            sentence_embed = sentence_output["last_hidden_state"]   # bsz x seq_len x hidden_size
            target_start_pos.unsqueeze_(1)
            target_end_pos.unsqueeze_(1)
            target_ids = torch.concat((target_start_pos, target_end_pos), dim=1).unsqueeze(1)
            target_embed = self.target_extractor(sentence_embed, target_ids).squeeze(1)  # 根据start end进行max_pooling
            target_embed = self.target_dropout(target_embed)

            # not use for complete classification
            target_logits = self.target_classifier(target_embed)

            frame_output = self.frame_encoder(
                input_ids=frame_input_ids,
                attention_mask=frame_attention_mask,
                token_type_ids=frame_token_type_ids,
                position_ids=frame_position_ids,
                head_mask=frame_head_mask,
            )
            # frame_output: CLS
            # framedef_embed = frame_output["last_hidden_state"][:,0,:]
            # framedef_embed = self.frame_dropout(framedef_embed)
            # frame_output: mean pooling 
            framedef_embed = frame_output["last_hidden_state"][:,1:,:]
            frame_mask = frame_attention_mask[:,1:]     # bsz x seq_len-1
            framedef_embed = framedef_embed.masked_fill(frame_mask.unsqueeze(-1).eq(0), 0).sum(dim=1)
            framedef_embed = framedef_embed / frame_mask.sum(dim=1, keepdim=True).float()
            framedef_embed = self.frame_dropout(framedef_embed)


            assert target_embed.shape == framedef_embed.shape, "target's shape is not equal to frame definition's shape"

            outputs = (target_embed, framedef_embed)
            
            # training  return: [loss, _, target, framedef]
            if labels is not None:
                assert self.config.train_mode == "contrastive_learning", KeyError("Loss func {} not exists".format(self.config.train_mode))
                loss = self.contrastive_loss(target_embed, framedef_embed, labels)
                print(f"      scl_loss: {loss:.5f}")
                
                outputs = (loss, torch.tensor(0, dtype=torch.float)) + outputs
            # evaluating or testing     return: [_, logits/_, target, framedef]
            else:# return similarity
                outputs = (torch.tensor(0, dtype=torch.float), torch.tensor(0, dtype=torch.float)) + outputs

            return outputs
    
    def contrastive_loss(self, target_embed, framedef_embed, labels):
        """now used, not wear away temperature effect"""
        labels = labels.contiguous().view(-1, 1)
        mask =  torch.eq(labels, labels.T).float()  # bsz x bsz

        target_embed = F.normalize(target_embed, dim=1)   # bsz x hidden_size
        framedef_embed = F.normalize(framedef_embed, dim=1)

        similarity = torch.div(
            torch.matmul(target_embed, framedef_embed.T),
            0.07)
        
        # min-max for numerical stability
        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1)
        loss = - mean_log_prob.mean()
        return loss


    def get_frame_embed(self, frame_input_ids=None, frame_attention_mask=None,
        frame_token_type_ids=None, frame_position_ids=None, frame_head_mask=None):

        frame_output = self.frame_encoder(
            input_ids=frame_input_ids,
            attention_mask=frame_attention_mask,
            token_type_ids=frame_token_type_ids,
            position_ids=frame_position_ids,
            head_mask=frame_head_mask,
        )
        # frame_output: CLS
        # framedef_embed = frame_output["last_hidden_state"][:,0,:]
        # framedef_embed = self.frame_dropout(framedef_embed)
        # frame_output: mean pooling 
        framedef_embed = frame_output["last_hidden_state"][:,1:,:]
        frame_mask = frame_attention_mask[:,1:]     # bsz x seq_len-1
        framedef_embed = framedef_embed.masked_fill(frame_mask.unsqueeze(-1).eq(0), 0).sum(dim=1)
        framedef_embed = framedef_embed / frame_mask.sum(dim=1, keepdim=True).float()
        framedef_embed = self.frame_dropout(framedef_embed)

        return framedef_embed