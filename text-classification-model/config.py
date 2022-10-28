import time
class DefaultConfig:

    load_model_path = time.strftime('checkpoints/model_' + '%m%d_%H:%M:%S.pth')
    batch_size = 8
    num_epochs = 1000
    learning_rate = 0.001
    input_size = 54

    hidden_size = 8
    output_size = 7
    lr_decay = 0.95
    weight_decay = 1e-4
    epoches = 1000
    print_freq = 100
