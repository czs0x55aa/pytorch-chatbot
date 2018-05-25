# coding=utf8
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class SearchState(object):
    def __init__(self, vocab, search_size=10, CUDA=False):
        self.TOKEN_GO = vocab.GO
        self.TOKEN_EOS = vocab.EOS
        self.search_size = search_size
        self.CUDA = CUDA

        self.beam_search_set = []   # 存储解码候选集
        self.cache_ids = [[]]   # 存储解码过程
        self.dec_input = Variable(torch.LongTensor([self.TOKEN_GO]))
        self.last_hidden = None
        self.last_prob = Variable(torch.FloatTensor([[0]])) # 解码概率

    def is_end(self):
        return self.search_size is 0

    def get_input(self):
        assert self.dec_input is not None
        return self.dec_input.cuda() if self.CUDA else self.dec_input

    def get_hidden(self):
        assert self.last_hidden is not None
        return self.last_hidden.cuda() if self.CUDA else self.last_hidden

    def get_prob(self):
        return self.last_prob.cuda() if self.CUDA else self.last_prob

    def get_result(self):
        return sorted(self.beam_search_set, key=lambda x: x['prob'], reverse=True)

    def update(self, step, log_outputs, hidden):
        vocab_size = log_outputs.size()[1]
        prob_update = log_outputs + self.get_prob().repeat(1, vocab_size)
        topv, topi = prob_update.data.topk(self.search_size, dim=1)

        row, col = topi.size()
        container = []
        for batch_idx in range(row):
            for col in range(self.search_size):
                container.append({
                    'prob': topv.cpu()[batch_idx][col],
                    'batch_idx': batch_idx,
                    'vocab_idx': topi.cpu()[batch_idx][col],
                })
        # log 概率 从小到大
        sorted_container = sorted(container, key=lambda item: item['prob'], reverse=True)[:self.search_size]
        next_prob, next_input, next_idx = [], [], []
        for item in sorted_container:
            if item['vocab_idx'] is self.TOKEN_EOS:
                # 解码完成
                self.beam_search_set.append({
                    'prob': item['prob'] / (len(self.cache_ids[item['batch_idx']]) + 1),
                    'ids': self.cache_ids[item['batch_idx']] + [item['vocab_idx']]
                })
                self.search_size -= 1
            else:
                next_prob.append(item['prob'])
                next_input.append(item['vocab_idx'])
                next_idx.append(item['batch_idx'])
        
        # 更新解码缓存
        self.cache_ids = [self.cache_ids[item['batch_idx']] + [item['vocab_idx']] for item in sorted_container if item['vocab_idx'] is not self.TOKEN_EOS]
        if self.search_size > 0:
            # 更新下一步的需要的输入
            self.last_prob = Variable(torch.FloatTensor([[p] for p in next_prob]))
            self.dec_input = Variable(torch.LongTensor(next_input))
            self.last_hidden = torch.stack([hidden[:, idx] for idx in next_idx], dim=1)


class BeamSearch(object):
    def __init__(self, task):
        self.task = task
        self.model = task.model
        self.enc_vocab = task.enc_vocab
        self.dec_vocab = task.dec_vocab
        self.CUDA = task.config['train']['CUDA']
        self.MAX_LEN = task.config['preproccess']['max_len']
        self.beam_size = task.config['test']['beam_size']
        self.antiLM = task.config['test']['antiLM']

    def decode(self, words, beam_size=0):
        self.model.eval()
        input_var = self.__get_var(self.enc_vocab.words2ids(words))
        beam_search_set = self.__beam_search(input_var, self.beam_size if beam_size is 0 else beam_size)
        self.model.train()
        return beam_search_set

    def __get_var(self, ids):
        data_var = Variable(torch.LongTensor([ids])).transpose(0, 1)
        return data_var.cuda() if self.CUDA else data_var

    def __beam_search(self, input_var, beam_size):
        # input_var size (length, 1)
        length = input_var.size(0)
        # Encoder
        # enc_outputs size (max_len, batch_size, hidden_size)
        # enc_hidden size (bi * n_layers, batch_size, hidden_size)
        enc_outputs, enc_hidden = self.model.encoder(input_var, [length])

        state = SearchState(self.enc_vocab, beam_size, CUDA=self.CUDA)
        state.last_hidden = enc_hidden[:self.model.decoder.n_layers]

        for step in range(self.MAX_LEN + 1):
            dec_input = state.get_input()
            last_hidden = state.get_hidden()

            search_size = dec_input.size()[0]
            # Decode
            dec_output, last_hidden = self.model.decoder(dec_input, last_hidden, enc_outputs.repeat(1, search_size, 1))
            prob_outputs = F.log_softmax(dec_output, dim=1)  # (search_size, vocab_size)

            state.update(step, prob_outputs, last_hidden)

            if state.is_end() is True:
                break
        
        if state.is_end() is False:
            # 存在超出解码长度的情况
            pass

        return state.get_result()

