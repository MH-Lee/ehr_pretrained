import torch
import torch.nn as nn


class RetainNN(nn.Module):
    def __init__(self, params):
        super(RetainNN, self).__init__()
        self.emb_layer = nn.Linear(in_features=params["n_diagnosis_codes"], out_features=params["hidden_size"])
        self.dropout = nn.Dropout(params["dropout_rate"])
        self.variable_level_rnn = nn.GRU(params["hidden_size"], params["hidden_size"])
        self.visit_level_rnn = nn.GRU(params["hidden_size"], params["hidden_size"])
        self.variable_level_attention = nn.Linear(params["hidden_size"], params["hidden_size"])
        self.visit_level_attention = nn.Linear(params["hidden_size"], 1)
        self.output_dropout = nn.Dropout(params["dropout_rate"])
        self.output_layer = nn.Linear(params["hidden_size"], params["n_labels"])
        self.var_hidden_size = params["hidden_size"]
        self.visit_hidden_size = params["hidden_size"]
        self.n_samples = params["batch_size"]
        self.reverse_rnn_feeding = False


    def forward(self, input, mask):
        """
        :param input:
        :param var_rnn_hidden:
        :param visit_rnn_hidden:
        :return:
        """
        # emb_layer: input(*): LongTensor of arbitrary shape containing the indices to extract
        # emb_layer: output(*,H): where * is the input shape and H = embedding_dim
        # print("size of input:")
        # print(input.shape)
        v = self.emb_layer(input)
        # print("size of v:")
        # print(v.shape)
        v = self.dropout(v)

        # GRU:
        # input of shape (seq_len, batch, input_size)
        # seq_len: visit_seq_len
        # batch: batch_size
        # input_size: embedding dimension
        #
        # h_0 of shape (num_layers*num_directions, batch, hidden_size)
        # num_layers(1)*num_directions(1)
        # batch: batch_size
        # hidden_size:
        if self.reverse_rnn_feeding:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(torch.flip(v, [0]))
            alpha = self.visit_level_attention(torch.flip(visit_rnn_output, [0]))
        else:
            visit_rnn_output, visit_rnn_hidden = self.visit_level_rnn(v)
            alpha = self.visit_level_attention(visit_rnn_output)
        visit_attn_w = torch.nn.functional.softmax(alpha, dim=0)

        if self.reverse_rnn_feeding:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(torch.flip(v, [0]))
            beta = self.variable_level_attention(torch.flip(var_rnn_output, [0]))
        else:
            var_rnn_output, var_rnn_hidden = self.variable_level_rnn(v)
            beta = self.variable_level_attention(var_rnn_output)
        var_attn_w = torch.tanh(beta)
        attn_w = visit_attn_w * var_attn_w

        c = torch.sum(attn_w * v, dim=0)
        c = self.output_dropout(c)
        output = self.output_layer(c)
        return output
