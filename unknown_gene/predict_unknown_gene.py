#!/usr/bin/env python

from __future__ import annotations

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import logging

from functools import partial
from pathlib import Path

from absl import flags, app
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorflow as tf
import umap

from ml_collections import config_flags
from pprint import pprint
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from lacss.train import Trainer, TFDatasetAdapter, VMapped

_CONFIG = config_flags.DEFINE_config_file("config")
_FLAGS = flags.FLAGS
flags.DEFINE_string("logpath", ".", "")

def loss_fn(batch, prediction):
    # L2 loss
    (gids, cnts), label = batch
    cnts_sum = cnts.sum(where=gids>=0)
    label = label / cnts_sum
    # prediction = jax.nn.softmax(prediction)
    loss = (prediction - label) ** 2
    loss = loss.sum() * 1000

    return loss

class Acc:
    def __init__(self, has_aux=False):
        self.l2_sum = 0
        self.cnts = 0
        self.states = []
        self.has_aux = has_aux

    def update(self, batch, prediction):
        if self.has_aux:
            prediction, aux = prediction
            self.states.append(aux["intermediates"]["state"][0])
        loss = jax.vmap(loss_fn)(batch, prediction)
        self.l2_sum += loss.mean()
        self.cnts += 1

    def compute(self):
        return self.l2_sum / self.cnts

def get_ds(config):
    linc_ids = [
        42779, 44419, 35393, 46963, 42780, 37033, 13320, 40206, 47022, 28615, 34008, 15060, 8452, 21878, 47102, 21877,
        25026, 23992, 635, 12555, 46975, 25022, 39607, 22533, 7390, 43382, 21854, 26083, 22165, 29254, 42777, 6943,
        35477, 39662, 14833, 38156, 22018, 9966, 47704, 21868, 2235, 21032, 36371, 721, 19087, 24189, 6325, 22122,
        2703, 27472, 16151, 15420, 5821, 1003, 40607, 8233, 7169, 42873, 23232, 38704, 16373, 32106, 627, 29405,
        34505, 46277, 74, 14433, 5747, 14747, 43166, 1004, 2292, 13304, 43809, 41842, 16211, 5438, 2889, 36800,
        35398, 38779, 26908, 29464, 32718, 13337, 22250, 37689, 44930, 23417, 9273, 38721, 35447, 41969, 9969, 1948,
        25019, 38642, 8237, 20162, 35957, 33263, 114, 33802, 1156, 632, 26170, 2297, 39343, 16463, 33160, 46969,
        12701, 14985, 42832, 41710, 12214, 45340, 43981, 42129, 41845, 26392, 43948, 36340, 21504, 42322, 42326, 28198,
        37016, 37038, 35939, 33041, 39125, 25021, 35887, 4474, 35158, 26917, 11467, 34963, 22233, 18581, 40420, 30835,
        41493, 43617, 26614, 32051, 32428, 9220, 16674, 25735, 33419, 37034, 36567, 34155, 36475, 38271, 12667, 19385,
        30703, 15415, 32898, 44033, 16147, 15054, 27485, 33247, 40847, 34585, 39490, 46121, 32499, 14287, 26214, 30067,
        24654, 20972, 3135, 8562, 34668, 890, 24638, 42330, 5741, 36376, 263, 15181, 36342, 39383, 11982, 10128,
        35424, 43593, 12123, 25091, 33388, 16718, 31736, 38965, 7423, 45306, 29499, 6618, 29594, 36562, 29553, 8552,
        15061, 14077, 43213, 44807, 25706, 25968, 15554, 26122, 22170, 30876, 38744, 44424, 36207, 38209, 9916, 38728,
        24252, 30321, 2121, 40453, 34586, 22478, 4261, 26346, 29381, 40177, 21846, 33265, 42177, 12673, 30780, 45668,
        27526, 13826, 12692, 41245, 3268, 12032, 16621, 16358, 16608, 25868, 8477, 15761, 14575, 36200, 3984, 15051,
        765, 15188, 24286, 3933, 38873, 37733, 23418, 30093, 34962, 7387, 1995, 33276, 461, 43945, 10066, 12377,
        29548, 39831, 16204, 29811, 12727, 43947, 9444, 29184, 27551, 43628, 35201, 40336, 31733, 2802, 40207, 1184,
        14909, 43610, 26587, 4532, 14720, 19356, 25013, 10015, 37037, 25352, 1185, 31214, 25023, 29260, 37615, 43654,
        311, 41701, 13221, 27058, 7603, 22291, 29418, 46786, 30011, 25112, 28201, 9612, 26615, 34386, 34969, 39926,
        3396, 35034, 15996, 37021, 38995, 29544, 4835, 40031, 15422, 32400, 29566, 43391, 5746, 3521, 43155, 16684,
        6358, 1796, 7174, 4798, 11609, 25603, 25646, 36809, 9177, 29547, 14987, 10116, 15794, 22035, 5734, 13441,
        40438, 42270, 34500, 10357, 14873, 18817, 37737, 2270, 34351, 36842, 45337, 9023, 41829, 28071, 21256, 20434,
        37028, 31699, 18053, 15006, 28699, 24886, 40278, 31826, 30263, 24451, 45616, 43651, 1955, 24849, 11338, 37160,
        9861, 16072, 29619, 25986, 46776, 41160, 11438, 22025, 5366, 11469, 3911, 40687, 21856, 26164, 30015, 15023,
        29293, 36579, 40489, 29993, 16218, 7297, 28349, 29395, 42257, 30277, 10534, 39287, 41751, 6260, 16616, 28700,
        12405, 2271, 12110, 16074, 4967, 12467, 14699, 28269, 20028, 30021, 11345, 1376, 15465, 8238, 30778, 13064,
        1914, 41635, 29579, 2063, 24442, 864, 43317, 45714, 10507, 28254, 45879, 8701, 717, 3479, 25017, 4683,
        40456, 31584, 31132, 16203, 9678, 22210, 43164, 23749, 7974, 22026, 41847, 42864, 43769, 27477, 43335, 19187,
        6210, 42009, 19832, 31871, 15984, 31700, 40936, 16970, 47990, 21533, 40904, 33035, 2518, 25770, 23996, 8006,
        7244, 28587, 34760, 36442, 35222, 39235, 26162, 8302, 8963, 37191, 9654, 6700, 11995, 22029, 36635, 7608,
        30273, 16534, 6741, 1037, 44111, 7992, 10873, 10327, 35515, 38883, 30758, 28764, 21844, 19988, 3158, 5735,
        25937, 29449, 30391, 35086, 27679, 46044, 16994, 3335, 21848, 784, 39460, 23577, 25636, 17202, 38676, 20476,
        44030, 33940, 26124, 3251, 9724, 43466, 38116, 28415, 43463, 13402, 26163, 43154, 18309, 1299, 43597, 38610,
        4843, 16291, 31917, 7587, 38131, 38263, 20005, 13908, 29545, 46062, 36940, 10196, 43100, 23273, 28112, 7153,
        2900, 18882, 5459, 43511, 45336, 40084, 4512, 31959, 43384, 37720, 29792, 24985, 18505, 40270, 23977, 5854,
        1294, 603, 20912, 22513, 25160, 9717, 16210, 30746, 35375, 4807, 15466, 11498, 41767, 2516, 7302, 17078,
        4463, 28815, 46345, 10498, 45072, 31188, 1870, 13399, 25473, 13619, 3133, 45020, 17096, 40603, 28541, 33437,
        39165, 27999, 31581, 22031, 45572, 6305, 16295, 6933, 39485, 6958, 24351, 41738, 20747, 30541, 25811, 15232,
        4471, 19838, 17954, 25351, 22067, 28037, 24775, 44854, 16933, 29801, 9702, 41287, 31968, 20140, 37193, 2781,
        20952, 28633, 25848, 37222, 39286, 20598, 35090, 12223, 26701, 34841, 27176, 37227, 9388, 40526, 13318, 20915,
        4386, 5752, 11028, 41465, 16510, 10256, 29636, 13396, 11178, 20918, 27955, 24979, 16624, 25737, 45191, 22299,
        7605, 44953, 3207, 34449, 42850, 3248, 4255, 1150, 45288, 11717, 4140, 4141, 23601, 40942, 24623, 42005,
        35524, 22066, 9967, 12198, 6064, 2566, 31538, 25818, 48701, 41161, 21987, 13401, 43159, 25731, 16153, 5965,
        24183, 26977, 18800, 8377, 8219, 43236, 7451, 42178, 33148, 23775, 1853, 46633, 10362, 36898, 6020, 18220,
        35371, 41941, 33790, 26049, 6610, 11093, 16298, 2493, 36896, 41045, 10234, 30659, 32695, 3673, 2508, 9088,
        19337, 29195, 43020, 26529, 16461, 41233, 25336, 12183, 10351, 26429, 17647, 40455, 34286, 1276, 32933, 25847,
        1484, 28286, 3342, 27678, 6154, 7698, 46028, 28434, 33789, 22040, 19259, 13744, 9965, 39502, 41604, 4825,
        25387, 13547, 40131, 21845, 28391, 8340, 9487, 39921, 28218, 30265, 19232, 24371, 14685, 25137, 40841, 40205,
        26695, 40042, 35119, 24564, 7879, 10417, 15293, 37162, 39054, 34269, 9715, 28061, 24659, 14311, 43400, 47933,
        13683, 28727, 3887, 15614, 33500, 3337, 27949, 5600, 25630, 9606, 27582, 24207, 31867, 26742, 34496, 23781,
        35092, 33311, 19256, 24734, 10924, 15082, 31817, 45006, 17365, 30658, 37858, 39140, 41745, 6904, 25090, 23046,
        22296, 45179, 24762, 1834, 14286, 26375, 34592, 25452, 6133, 44285, 23434, 23225, 24212, 11502, 45642, 18128,
        4597, 37762, 33269, 43342, 39481, 2289, 22303, 39431, 4259, 18815, 40938, 24671, 31586, 33067, 1042, 33295,
        46257, 39069, 1556, 7120, 14796, 10217, 8450, 27774, 28397, 10104, 1918, 34027, 39149, 18416, 26525, 2527,
        6206, 30766, 9066, 39655, 30550, 11076, 45269, 3222, 2803, 27928, 24738, 7518, 24959, 20901, 24922, 39854,
        25223, 27574, 41418, 22051, 2265, 12251, 27665, 42570, 10820, 298, 11465, 21067, 31796, 16438, 1273, 25941,
        4315, 21969, 32115, 17295, 1924, 18026, 7888, 37163, 23730, 44891, 9877, 37116, 30669, 3411, 36929, 2425,
        1540, 15421, 16392, 25729, 27992, 15291, 27139, 18501, 40032, 28441, 9428, 19268, 13941, 15506, 9657, 3262,
        17132, 6853, 27446, 39216, 16177, 1033, 11858, 23648, 30800, 9949, 39162, 28610, 8853, 32429, 15250, 7596,
        7993, 23267, 7170, 25109, 7401, 12648, 22021, 15975, 6207, 24249, 25935, 7880, 17958, 3457, 27617, 8259,
        7609, 17783, 503, 11680, 17086, 4890, 9268, 24661, 27664, 17881, 42233, 39205, 25989, 13957, 16579, 12489,
        45256, 13955, 25702, 13450, 43299, 763, 16601, 10945, 397, 22586, 1800, 1572, 45205, 30039, 6047, 25524,
        4760, 27113, 35211, 36476, 23988, 27160, 28129, 14787, 34398, 23456, 41612, 27675, 34170, 25810, 26403, 12209,
    ]
        
    padding = config.dataset.get("padding", 4096)
    ds_location = config.dataset.get(
        "path",
        "../tome.ds"
    )

    with open(Path(ds_location)/"metadata") as f:
        metadata = json.load(f)

    is_coding = tf.constant(np.asarray(metadata["gene_type"]) == "protein_coding")
    coding_gene_ids = tf.cumsum(tf.cast(is_coding, tf.int32)) - 1

    # create a map where the target genes are mapped to 0-1024 and the rest are mapped to -1
    linc_map = np.zeros([len(metadata["gene_type"])], dtype=int)
    linc_map[linc_ids] = 1
    m2 = linc_map 
    linc_map = np.cumsum(linc_map) - 1
    linc_map = np.where(m2, linc_map, -1)
    linc_map = tf.constant(linc_map)


    def prune(idx, cnts, tp):
        sel_coding = tf.gather(is_coding, idx)
        idx_coding = tf.gather(coding_gene_ids, idx[sel_coding])
        idx_coding = idx_coding[:padding]
        cnts_coding = cnts[sel_coding][:padding]

        target_idx = tf.gather(linc_map, idx)
        sel_target = target_idx >= 0
        target_idx = target_idx[sel_target]
        target_cnts = cnts[sel_target]
        targets = tf.scatter_nd(target_idx[:, None], target_cnts, [1024])

        return (idx_coding, cnts_coding), targets

    ds = (
        tf.data.Dataset.load(ds_location)
        .map(prune)
    )

    ds_train = (
        ds.enumerate()
        .filter(lambda n,x: n%5 != 0)
        .map(lambda n,x: x)
        .padded_batch(
            config.train.batchsize,
            padded_shapes=(
                ([padding], [padding]), [1024],
            ),
            padding_values=-1,
        )    
        .repeat()
        .prefetch(1)
    )    

    ds_val = (
        ds.enumerate()
        .filter(lambda n,x: n%5 == 0)
        .map(lambda n, x: x)
        .padded_batch(
            config.train.batchsize,
            padded_shapes=(
                ([padding], [padding]), [1024],
            ),
            padding_values=-1,
            drop_remainder=True,
        )
        .prefetch(1)
    ) 
    return ds_train, ds_val, metadata

def main(_):
    config = _CONFIG.value
    pprint(config)
    logpath = Path(_FLAGS.logpath)
    seed = config.train.get("seed", 42)

    for i in range(config.get("num_runs", 3)):
        run(config, logpath/str(i), seed)
        seed = seed + 1

def run(config, logpath, seed):
    logpath.mkdir(parents=True, exist_ok=True)

    logging.info(f"Logging to {logpath.resolve()}")

    ds_train, ds_val, metadata = get_ds(config)

    lr = config.train.lr
    model = config.model.type(**config.model.config)
    freeze_embedding = config.model.get("freeze_embed", False)

    trainer = Trainer(
        model = model,
        optimizer = optax.adamw(lr),
        losses = loss_fn,
        strategy= VMapped,
        seed = seed,
    )

    train_it = trainer.train(
        TFDatasetAdapter(ds_train), 
        rng_cols=["dropout"], 
        training=True, 
    )

    if freeze_embedding:
        frozen = jax.tree_util.tree_map_with_path(
            lambda p, _: jax.tree_util.DictKey("Embed_0") in p, train_it.parameters,
        )
        train_it = trainer.train(
            TFDatasetAdapter(ds_train), 
            rng_cols=["dropout"], 
            frozen=frozen,
            training=True,            
        )

    embed = config.model.get("embed", None)
    if embed is not None:
        train_it.parameters["Embed_0"]["embedding"] = jnp.asarray(embed)

    # pb = tqdm(range(config.train.train_steps))
    pb = range(config.train.train_steps)
    for step in pb:
        # peek = train_it.data.peek()

        pred = next(train_it)
        # desc = repr(train_it.loss_logs)
        # if "nan" in desc:
        #     print(jnp.any(jnp.isnan(pred)))
        #     # print(peek[0])
        #     print(peek[1].sum(axis=-1) == 0)
        #     exit(1)

        # pb.set_description(desc)
        if (step + 1) % config.train.validation_interval == 0:
            print(f"step: {step+1}")
            print(train_it.loss_logs)

            train_it.reset_loss_logs()

            print(trainer.compute_metrics(
                TFDatasetAdapter(ds_val),
                Acc(),
                dict(params=train_it.parameters),
            ))

    # metric = Acc(True)
    # print(trainer.compute_metrics(
    #     TFDatasetAdapter(ds_val),
    #     metric,
    #     dict(params=train_it.parameters),
    #     mutable="intermediates",
    # ))
    # xs = np.concatenate(metric.states)
    # ct = np.concatenate(metric.y_true)

    # reducer = umap.UMAP()
    # reduced = reducer.fit_transform(xs[::10])

    # plt.figure(figsize=(20,20))
    # f = plt.scatter(reduced[:, 0], reduced[:, 1], s=0.1, c=ct[::10])
    # handles, labels = f.legend_elements(num=68, )
    # plt.gca().legend(handles, metadata["cell_types"])

    # plt.savefig(str(logpath/"cell_states.png"))
    # plt.close()

    cp = ocp.StandardCheckpointer()
    cp.save(
        (logpath/"checkpoint").absolute(), 
        args=ocp.args.StandardSave(train_it),
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(main)
