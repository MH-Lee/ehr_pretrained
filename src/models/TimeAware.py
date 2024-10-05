import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class TimeLSTM(nn.Module):
    def __init__(self, n_diagnosis_codes, hidden_size, device=torch.device('cude:0'), bidirectional=False):
        # assumes that batch_first is always true
        super(TimeLSTM,self).__init__()
        self.hidden_size = hidden_size
        self.n_diagnosis_codes = n_diagnosis_codes
        self.device = device
        self.W_all = nn.Linear(n_diagnosis_codes, hidden_size*4)
        self.U_all = nn.Linear(hidden_size, hidden_size*4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.bidirectional = bidirectional
        emb = nn.Embedding(n_diagnosis_codes, hidden_size, padding_idx=0)
        self.emb = emb

    def forward(self, inputs_list, reverse=False):
        inputs = inputs_list[0]
        timestamps = 1 / torch.log(inputs_list[2] + 2.7183)
        mask = inputs_list[1]
        inputs = (self.emb(inputs) * mask.unsqueeze(-1)).sum(dim=2)
        b, seq, hid = inputs.size()
        h = torch.zeros(b, hid, device=self.device, requires_grad=False)
        c = torch.zeros(b, hid, device=self.device, requires_grad=False)

        outputs = []
        for s in range(seq):
            c_s1 = F.tanh(self.W_d(c))
            c_s2 = c_s1 * timestamps[:,s:s+1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:,s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = F.sigmoid(f)
            i = F.sigmoid(i)
            o = F.sigmoid(o)
            c_tmp = F.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * F.tanh(c)
            outputs.append(h)
            
        if reverse:
            outputs.reverse()
        outputs = torch.stack(outputs, 1)
        return outputs