# coding=utf8
import sys
import re
from config_default import *
import utils
from utils import Task
from modules.beam_search import BeamSearch

replacement_patterns = [
    (r'[wW]{1}on(\s?)\'(\s?)t', 'will not'),
    (r'[cC]{1}an(\s?)\'(\s?)t', 'cannot'),
    (r'[iI]{1}(\s?)\'(\s?)m', 'i am'),
    (r'ain(\s?)\'(\s?)t', 'is not'),
    (r'(\w+)(\s?)\'(\s?)ll', '\g<1> will'),
    (r'(\w+)n(\s?)\'(\s?)t', '\g<1> not'),
    (r'(\w+)(\s?)\'(\s?)ve', '\g<1> have'),
    (r'(\w+)(\s?)\'(\s?)s', '\g<1> is'),
    (r'(\w+)(\s?)\'(\s?)re', '\g<1> are'),
    (r'(\w+)(\s?)\'(\s?)d', '\g<1> would'),
    (r' o(\s?)\'(\s?)clock', ' of the clock'),
]

class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
        self._WORD_SPLIT = re.compile("([.,!?\"':;)(])")

    def replace(self, text):
        # 过滤特殊字符
        s = re.sub(r"[^a-zA-Z\d',.!?$:-]+", r" ", text)
        # 拆分缩写
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        # 拆分标点
        text = []
        for fragment in s.split():
            text.extend(re.split(self._WORD_SPLIT, fragment))
        return ' '.join(text)

class LoopBot(object):
    def __init__(self, debug=False):
        self.name = 'Bot'
        self.replacer = RegexpReplacer()
        self.debug = debug

    def launch(self):
        while True:
            user_input = input('me: ')
            if user_input == 'exit':
                break
            trim_input = self.trim(user_input)
            if self.debug:
                print(f'user_input: {user_input}')
                print(f'trim_input: {trim_input}')
            response = self.service(trim_input)
            self.print(response)

    def print(self, response):
        if isinstance(response, str):
            print(f'{self.name}: {response}\n')

    def trim(self, input):
        text = self.replacer.replace(input)
        return [word.lower() for word in text.split()]

    def service(self, input):
        return 'hello world.'


class ChatBot(LoopBot):
    def __init__(self, task, debug=False):
        super(ChatBot, self).__init__(debug)
        self.name = 'Bot'
        self.task = task
        self.beam_search = BeamSearch(task)

    def service(self, input):
        response = self.beam_search.decode(input)
        response_group = []
        for resp in response:
            response_group.append({
                'text': ' '.join(self.task.dec_vocab.ids2word(resp['ids'])),
                'prob': resp['prob']
            })
        return response_group

    def print(self, response):
        if isinstance(response, list):
            for resp in response[:5]:
                print(f"{self.name}: {resp['text']} score:{resp['prob']}")
            print('\n')


if __name__ == '__main__':

    if len(sys.argv) > 2:
        task = Task(config)
        task.load(mode='test', ckpt_path=sys.argv[1], model_name=sys.argv[2])
        chatbot = ChatBot(task)
    else:
        chatbot = LoopBot()
    
    chatbot.launch()
    

    
