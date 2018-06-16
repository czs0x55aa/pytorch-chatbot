# coding=utf8

config = {
    'dataset': {
        'path': '.',
        'dir': 'data',
        'name': 'train.txt'
    },
    'preproccess': {
        'lang': 'en',
        'min_count': 2,
        'min_len': 1,
        'max_len': 20,
        'n_test_batch': 10
    },
    'token': {
        'PAD': 0,
        'GO': 1,
        'EOS': 2,
        'UNK': 3
    },
    'train': {
        'CUDA': False,
        'batch_size': 100,
        'n_epochs': 6,
        'optim': 'adam',
        'learning_rate': 0.001,
        'max_grad_norm': 5,
        'teacher_forcing_ratio': 1.0,
        'print_every': 100,
        'checkpoint': False,
        'silence': False
    },
    'model': {
        'embedding_size': 500,
        'enc_layers': 1,
        'dec_layers': 1,
        'bidirectional': True,
        'hidden_size': 500,
        'dropout': 0.15,
        'param_init': 0.1,
        'attn_type': 'general'
    },
    'test': {
        'autotest': False,
        'beamsearch': True,
        'beam_size': 50,
        'antiLM': 0.5
    }
}
