# coding=utf8
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src_var, tgt_var, src_lens, teacher_forcing_ratio=1):

        enc_outputs, enc_hidden = self.encoder(src_var, src_lens)

        dec_length, batch_size = tgt_var.size()

        # store all decoder outputs
        all_outputs = []
        dec_hidden = enc_hidden[:self.decoder.n_layers]
        dec_input = None
        dec_output = None
        for t in range(dec_length):

            # select real target or decoder output
            teacher_forcing = random.random() < teacher_forcing_ratio
            if teacher_forcing is True or dec_output is None:
                dec_input = tgt_var[t]
            else:
                prob_output = F.log_softmax(dec_output, dim=1)
                _, topi = prob_output.data.topk(1, dim=1)
                dec_input = Variable(topi.squeeze(1))

            dec_output, dec_hidden = self.decoder(dec_input, dec_hidden, enc_outputs)
            all_outputs.append(dec_output)

        outputs = torch.stack(all_outputs)
        return outputs, dec_hidden

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1, dropout=0.1, bidirectional=True):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.input_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_var, input_lens=None, hidden=None):
        # embedded size (max_len, batch_size, hidden_size)
        embedded = self.embedding(input_var)
        # embedded = self.input_dropout(embedded)

        packed_emb = embedded
        if input_lens is not None:
            packed_emb = pack_padded_sequence(packed_emb, input_lens)

        outputs, enc_hidden = self.gru(packed_emb, hidden)

        if input_lens is not None:
            outputs = pad_packed_sequence(outputs)[0]

        if self.num_directions == 2:
            # sum bidirectional outputs
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # outputs size (max_len, batch_size, hidden_size)
        # hidden size (bi * n_layers, batch_size, hidden_size)
        return outputs, enc_hidden


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1, dropout=0.1, attn_type='dot'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.attn_type = attn_type
        assert attn_type in ['dot', 'general', 'concat', 'none']

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedded_dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        if attn_type != 'none':
            self.attn = Attn(attn_type, hidden_size)
            self.concat = nn.Linear(hidden_size * 2, hidden_size)

        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_var, last_hidden, enc_ouputs, real_input_var=None):
        # input_var size (batch_size,)
        # last_hidden size (n_layers, batch_size, hidden_size)
        # enc_ouputs size (max_len, batch_size, hidden_size)

        batch_size = input_var.size(0)
        # embedded size (1, batch_size, hidden_size)
        embedded = self.embedding(input_var).unsqueeze(0)
        if real_input_var is not None:
            embedded = (
                embedded + self.embedding(real_input_var).unsqueeze(0)) / 2
        # embedded = self.embedded_dropout(embedded)

        # rnn_output size (1, batch_size, hidden_size)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        if self.attn_type != 'none':
            # attn_weights size (batch_size, 1, max_len)
            attn_weights = self.attn(rnn_output, enc_ouputs)
            # context size (batch_size, 1, hidden_size) = (batch_size, 1, max_len) * (batch_size, max_len, hidden_size)
            context = attn_weights.bmm(enc_ouputs.transpose(0, 1))
            # concat_input size (batch_size, hidden_size * 2)
            concat_input = torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1)
            rnn_output = self.concat(concat_input)
        else:
            rnn_output = rnn_output.squeeze(0)

        # output size (batch_size, vocab_size)
        output = self.out(F.tanh(rnn_output))
        return output, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(
                self.hidden_size, self.hidden_size, bias=False)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2,
                                  self.hidden_size, bias=False)
            self.v = nn.Parameter(weight_init.xavier_uniform(
                torch.FloatTensor(1, self.hidden_size)))

    def forward(self, hidden, encoder_outputs):
        # attn_energies size: (batch_size, max_len)
        attn_energies = self.batch_score(hidden, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    # faster
    def batch_score(self, hidden, encoder_outputs):
        # encoder_outputs size (max_len, batch_size, hidden_size)
        energy = None
        if self.method == 'dot':
            # encoder_outputs size (batch_size, hidden_size, length)
            encoder_outputs = encoder_outputs.permute(1, 2, 0)
            energy = torch.bmm(hidden.transpose(
                0, 1), encoder_outputs).squeeze(1)
        elif self.method == 'general':
            length, batch_size, _ = encoder_outputs.size()
            energy = self.attn(encoder_outputs.view(-1, self.hidden_size)
                               ).view(length, batch_size, self.hidden_size)
            energy = torch.bmm(hidden.transpose(
                0, 1), energy.permute(1, 2, 0)).squeeze(1)
        elif self.method == 'concat':
            length, batch_size, _ = encoder_outputs.size()
            attn_input = torch.cat(
                (hidden.repeat(length, 1, 1), encoder_outputs), dim=2)
            energy = self.attn(attn_input.view(-1, 2 * self.hidden_size)
                               ).view(length, batch_size, self.hidden_size)
            energy = torch.bmm(self.v.repeat(batch_size, 1, 1),
                               F.tanh(energy.permute(1, 2, 0))).squeeze(1)
        return energy
