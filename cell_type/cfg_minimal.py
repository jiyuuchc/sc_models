from __future__ import annotations

import ml_collections
import numpy as np
import optax

from sc_models import SCLinear

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "celltype_random_embedding"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "/home/FCAM/jyu/datasets/tome.ds"

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.batchsize = 128
    config.train.train_steps = 60000
    config.train.validation_interval = 5000

    lr = optax.piecewise_constant_schedule(0.0005, {50000: 0.1})
    config.train.lr = lr

    config.model = ml_collections.ConfigDict()
    config.model.type = SCLinear
    config.model.config = ml_collections.ConfigDict()
    config.model.embed = np.random.normal(size=(21973, 256))
    config.model.freeze_embed = True

    config.num_runs = 3

    return config
