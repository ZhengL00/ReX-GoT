def configure_scheduler(optimizer, num_training_steps, args):
    """
    配置学习率调度器
    :param optimizer: 已配置好的优化器
    :param num_training_steps: 总的训练步数
    :param args: 包含调度器相关参数的对象，如warmup_steps, warmup_ratio, lr_scheduler_type等
    :return: 配置好的学习率调度器实例
    """
    if not hasattr(args, "warmup_ratio"):
        args.warmup_ratio = 0.1  # 如果没有设置warmup_ratio，默认设置为0.1
    if not hasattr(args, "lr_scheduler_type"):
        args.lr_scheduler_type = "linear"  # 如果没有设置lr_scheduler_type，默认设置为linear

    warmup_steps = (
        args.warmup_steps
        if hasattr(args, "warmup_steps") and args.warmup_steps > 0
        else math.ceil(num_training_steps * args.warmup_ratio)
    )
    try:
        lr_scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    except ValueError as ve:
        raise ValueError(f"Error configuring scheduler: {ve}") from ve
    return lr_scheduler
