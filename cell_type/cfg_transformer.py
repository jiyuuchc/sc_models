from __future__ import annotations

import ml_collections
import optax

from sc_models import SCTransformer

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "celltype_transformer_model"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "/home/FCAM/jyu/datasets/tome.ds"
    config.dataset.padding = 2048

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.batchsize = 16
    config.train.train_steps = 240000
    config.train.validation_interval = 20000

    lr = optax.piecewise_constant_schedule(0.0001, {50000: 0.1})
    config.train.lr = lr

    config.model = ml_collections.ConfigDict()
    config.model.type = SCTransformer
    config.model.config = ml_collections.ConfigDict()

    config.num_runs = 1

    return config
