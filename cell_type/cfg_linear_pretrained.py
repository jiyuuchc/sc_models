from __future__ import annotations

import ml_collections
import optax
import numpy as np

from sc_models import SCLinear

def get_config():
    config = ml_collections.ConfigDict()
    config.name = "celltype_linear_model"

    config.dataset = ml_collections.ConfigDict()
    config.dataset.path = "/home/FCAM/jyu/datasets/tome.ds"

    config.train = ml_collections.ConfigDict()
    config.train.seed = 4242
    config.train.batchsize = 128
    config.train.train_steps = 70000
    config.train.validation_interval = 5000
    lr = optax.piecewise_constant_schedule(0.0001, {50000: 0.1})
    config.train.lr = lr

    config.model = ml_collections.ConfigDict()
    config.model.type = SCLinear
    config.model.config = ml_collections.ConfigDict()

    config.model.embed = np.load("gene_embedding.npz")["data"]
    config.train.freeze_embedding = True

    config.num_runs = 3

    return config
