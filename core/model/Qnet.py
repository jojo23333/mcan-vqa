import torch, copy
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from core.model.net_utils import FC, MLP, LayerNorm
from core.model.mca import MCA_ED
from typing import Optional, List

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class TransformerDecoderLayer(nn.Module):
    """
        Modified from pytorch implementation, do normalization for input but not output
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None,
                # query_pos: Optional[Tensor] = None):
        #print("tgt:", tgt.shape)
        #print("memory", memory.shape)
        tgt = self.norm1(tgt)
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        return tgt


class DecoderLayerGCN(nn.Module):
    """
        Modified from pytorch implementation, do normalization for input but not output
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer
    """
    def __init__(self, d_model, nhead, num_queries=3129, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayerGCN, self).__init__()
        self.gcn = GCN(num_state=d_model, num_node=num_queries)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None,
                # query_pos: Optional[Tensor] = None):
        #print("tgt:", tgt.shape)
        #print("memory", memory.shape)
        tgt = self.norm1(tgt)
        tgt2 = self.gcn(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        return tgt


class DecoderLayerNoSelfAtt(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayerNoSelfAtt, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
                # pos: Optional[Tensor] = None,
                # query_pos: Optional[Tensor] = None):
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm3(tgt)
        return tgt


class Qclassifier(nn.Module):
    def __init__(self, __C, embedding):
        super(Qclassifier, self).__init__()

        # TODO:
        NUM_DECODER_LAYER = 2
        if __C.DECODER_CLASSIFIER == 'GCN':
            decoder_layer = DecoderLayerGCN(d_model = __C.HIDDEN_SIZE,
                                            nhead = 4,
                                            dim_feedforward = 512)
        elif __C.DECODER_CLASSIFIER == 'NoSelfAtt':
            decoder_layer = DecoderLayerNoSelfAtt(d_model = __C.HIDDEN_SIZE,
                                                    nhead = 4,
                                                    dim_feedforward = 512)
        self.decoder_layers_img = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(NUM_DECODER_LAYER)])
        self.decoder_layers_ques = nn.ModuleList([copy.deepcopy(decoder_layer) for i in range(NUM_DECODER_LAYER)])

        # TODO: final linear classifier part
        self.head_norm = nn.LayerNorm(__C.HIDDEN_SIZE)
        self.head = nn.Linear(__C.HIDDEN_SIZE, 1)

    def forward(self, ans_feat, ques_feat, img_feat, ques_mask, img_mask):
        """
        ans_feat should be (NUM OF ANSWERS, LEN OF ANSWERS)
        """
        # expand ans_feat
        batch_size = ques_feat.shape[0]
        ans_feat = ans_feat.repeat(batch_size, 1, 1)
        # print(ans_feat.shape, img_feat.shape, ques_feat.shape)

        # decoder layers
        tgt_img = ans_feat.transpose(0, 1)
        for layer in self.decoder_layers_img:
            tgt_img = layer(tgt=tgt_img, 
                            memory=img_feat.transpose(0, 1),
                            memory_key_padding_mask=img_mask.squeeze())

        tgt_ques = ans_feat.transpose(0, 1)
        for layer in self.decoder_layers_ques:
            tgt_ques = layer(tgt=tgt_ques,
                             memory=ques_feat.transpose(0, 1),
                             memory_key_padding_mask=ques_mask.squeeze())
        
        # TODO: fuse from image and ques output
        tgt = tgt_ques + tgt_img
        tgt = tgt.transpose(0, 1)

        # TODO: final predict part
        scores = self.head(self.head_norm(tgt)) # (batch, num_queries)
        scores = torch.sigmoid(scores.squeeze())
        return scores


class Net_QClassifier(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net_QClassifier, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.lstm_ans = torch.nn.LSTM(
            input_size = __C.WORD_EMBED_SIZE,
            hidden_size = __C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)
        self.head = Qclassifier(__C, self.embedding)
        self.answer_size = answer_size


    def forward(self, input_dict):#img_feat, ques_ix, ans_ix):
        img_feat = input_dict['img_feat']
        ques_ix = input_dict['ques_ix']
        ans_ix = input_dict['ans_ix']

        assert ans_ix.size()[1] == self.answer_size

        # Make mask
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # get ans_feat
        self.lstm_ans.flatten_parameters()
        ans_feat = self.embedding(ans_ix[0])
        ans_feat, _ = self.lstm_ans(ans_feat)
        
        # only take the last output of lstm, could be changed later
        ans_feat = ans_feat[None,:,-1,:]

        pred = self.head(ans_feat, lang_feat, img_feat, lang_feat_mask, img_feat_mask)

        return pred, None


# Masking
def make_mask(feature):
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
