class DefautConfig:

    #data
    import time
    mnist_path = 'datasets/mnist'
    svhn_path = 'datasets/svhn'
    mode = 'train'
    model_path = time.strftime('checkpoints/model-' + '%m%d-%H:%M:%S')
    log_step = 10
    sample_step = 500
    sample_path = 'outputs/'



    #model parameter
    img_size = 32
    conv_dim = 64
    num_classes = 10
    num_layers = 3
    input_dim1 = 1
    input_dim2 = 3

    #training hyperparamater
    train_iter = 40000
    batch_size = 64
    num_workers = 2
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    use_reconst_loss = True


