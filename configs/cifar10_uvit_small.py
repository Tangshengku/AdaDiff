import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.exit_threshold = 0.9

    config.train = d(
        n_steps=200000,
        batch_size=512,
        mode='uncond',
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.0004,
        weight_decay=0.03,
        betas=(0.99, 0.999),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=2500
    )

    config.nnet = d(
        name='uvit',
        img_size=32,
        patch_size=2,
        embed_dim=512,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=-1,
    )

    config.dataset = d(
        name='cifar10',
        path='assets/datasets/cifar10',
        random_flip=True,
    )

    config.sample = d(
        sample_steps=1000,
        n_samples=50000,
        mini_batch_size=1,
        algorithm='euler_maruyama_sde',
        path=""
    )

    return config
