# pytorch-chatbot
Seq2Seq chatbot implement using PyTorch  
Feature: Seq2Seq + Beam Search + antiLM

## Requirements
- Python3
- Pytorch 0.3

## Corpus
- [DailyDialog](http://www.aclweb.org/anthology/I17-1099)

## Usage
### Training
```python
python train.py
```
### Test
```python
python console python console.py ./ckpt model
```
## Beam Search Example:
```
me: hi .  
Bot: how can i help you ? score:-0.66  
Bot: where are you going to go ? score:-0.66  
Bot: i am sorry to hear that . what can i do for you ? score:-0.67  
Bot: where are you going ? score:-0.68  
Bot: how are you going to do that ? score:-0.72  

me: what's your name ?  
Bot: my name is sam . score:-0.46  
Bot: my name is mona white . score:-0.53  
Bot: my name is james . score:-0.57  
Bot: my name is zhuang lingy . how are you , miss kelly ? score:-0.57  
Bot: my name is zhuang lingy . how are you ? score:-0.61  

me: how old are you ?  
Bot: i am twenty-five years old . score:-0.85  
Bot: i am not sure . what about you ? score:-0.89  
Bot: i am going to have a picnic with my friends . score:-0.96  
Bot: i am going to buy a birthday party for you . score:-0.97  
Bot: 5 years old . score:-0.98  
```

## Reference
- [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055.pdf)
- [seq2seq-translation.ipynb](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
- [Pytorch Documentation](https://pytorch.org/docs/0.3.0/)