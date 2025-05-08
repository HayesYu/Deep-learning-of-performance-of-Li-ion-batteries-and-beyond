import torch

# 加载现有的 checkpoint
checkpoint = torch.load('/home/user1/code/transfer_learning_model/checkpoint.pth.tar')

# 手动修改 checkpoint，添加 momentum
if 'optimizer' in checkpoint:
    optimizer = checkpoint['optimizer']

    # 如果需要，可以初始化 SGD 优化器并设置默认的 momentum 参数
    default_momentum = 0.9  # 设置默认的 momentum 参数
    for param_group in optimizer['param_groups']:
        if 'momentum' not in param_group:  # 如果没有 momentum，就添加
            param_group['momentum'] = default_momentum

    default_dampening = 0.0  # 通常 SGD 的 dampening 默认为 0
    for param_group in optimizer['param_groups']:
        if 'dampening' not in param_group:  # 如果没有 dampening，就添加
            param_group['dampening'] = default_dampening
    
    # 将修改后的状态字典重新存入 checkpoint
    checkpoint['optimizer'] = optimizer


# 保存修改后的 checkpoint
torch.save(checkpoint, 'checkpoint_new.pth.tar')
