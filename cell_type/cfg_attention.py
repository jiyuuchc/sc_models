from __future__ import annotations

import ml_collections
import optax

from sc_models import SCAtt

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "celltype_attention_model"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "/home/FCAM/jyu/datasets/tome.ds"

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.batchsize = 64
    config.train.train_steps = 140000
    config.train.validation_interval = 10000

    lr = optax.piecewise_constant_schedule(0.0001, {50000: 0.1})
    config.train.lr = lr

    config.model = ml_collections.ConfigDict()
    config.model.type = SCAtt
    config.model.config = ml_collections.ConfigDict()

    config.num_runs = 3

    return config
