from ARC_2.model import ArcModel
from CDSSM.model import CDSSM
from MVLSTM.model import MVLSTM
import time



class DefaultConfig:

    data_path = {'train':'', 'valid':'', 'test': ''}
    batch_size = 8
    epoch = 200
    lr = 0.001
    weight_decacy = 0.0001
    output_path = time.strftime('checkpoints/model_' + '%m%d_%H:%M:%S.pth')
    print_fq = 50

    def build_mvlstm(self):
        args = {'num_words':200, 'embed_dim': 64, 'hidden_size':32, 'num_layers':1, 'num_classes':2, 'query_length':10, 'k':5}
        model = MVLSTM(args)
        return model

    def build_CDSSM(self):
        WORD_DEPTH = 3000
        CONV_DIM = 1024
        k = 300
        L = 128
        FILTER_LENGTH = 3
        p  = 0.4
        model = CDSSM(WORD_DEPTH, CONV_DIM,k, L, FILTER_LENGTH, p)
        return model


    def build_Arc2(self):
        args = {'word_num':3000, 'embedding_dim':128, 'query_length':35, 'conv_dim':35, 'filter_length':3}
        model = ArcModel(args)
        return model






