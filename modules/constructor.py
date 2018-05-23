# coding=utf8

from component import Encoder, Decoder, Seq2Seq

def make_encoder(cf, vocab_size):

    encoder = Encoder(
        vocab_size,
        embedding_size=cf['embedding_size'],
        hidden_size=cf['hidden_size'],
        n_layers=cf['enc_layers'],
        dropout=cf['dropout'],
        bidirectional=cf['bidirectional']
    )
    return encoder

def make_decoder(cf, vocab_size):

    decoder = Decoder(
        vocab_size,
        embedding_size=cf['embedding_size'],
        hidden_size=cf['hidden_size'],
        n_layers=cf['dec_layers'],
        dropout=cf['dropout'],
        attn_type=cf['attn_type']
    )
    return decoder

def make_base_model(model_config, enc_vocab_size, dec_vocab_size):
    encoder = make_encoder(model_config, enc_vocab_size)
    decoder = make_decoder(model_config, dec_vocab_size)
    model = Seq2Seq(encoder, decoder)

    # 初始化参数
    param_init = model_config['param_init']
    if param_init != 0:
        for p in model.parameters():
            p.data.uniform_(-param_init, param_init)
            
    return model
