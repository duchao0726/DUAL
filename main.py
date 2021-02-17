from __future__ import division
import argparse

parser = argparse.ArgumentParser()
# Basic & Model parameters
parser.add_argument("-gpu", type=str, default="0")
parser.add_argument("-message", type=str, default="")
parser.add_argument("-loglv", type=str, default="debug")
parser.add_argument("-method", type=str, default="DNN")
parser.add_argument("-dataset", type=str, default="Books")
parser.add_argument("-seed", type=int, default=12345)
parser.add_argument("-model_path", type=str, default=None)
parser.add_argument("-dims", type=str, default="200,80")
parser.add_argument("-optimizer", type=str, default="Adam")
parser.add_argument("-batch_size", type=int, default=64)
parser.add_argument("-maxlen", type=int, default=100)
parser.add_argument("-lr", type=float, default=1e-3)
parser.add_argument("-lr_decay", type=float, default=0.5)
parser.add_argument("-momentum", type=float, default=0.9)
parser.add_argument("-epochs", type=int, default=2)
parser.add_argument("-log_every", type=int, default=100)
parser.add_argument("-viz_every", type=int, default=200)
parser.add_argument("-model_every", type=int, default=1e10)
parser.add_argument("-output", type=str, default="output")
# SVGP parameters
parser.add_argument("-n_ind", type=int, default=200)
parser.add_argument("-lengthscale", type=float, default=2.0)
parser.add_argument("-amplitude", type=float, default=0.3)
parser.add_argument("-diag_cov", dest="diag_cov", action="store_true")
parser.set_defaults(diag_cov=False)
parser.add_argument("-jitter", type=float, default=1e-4)
parser.add_argument("-prior_mean", type=float, default=0.0)
parser.add_argument("-temp", type=float, default=1e-6)
parser.add_argument("-km_coeff", type=float, default=0.1)
args = parser.parse_args()

# ----------------------------------------------------------------

import os

# set environment variables
os.environ["PYTHONHASHSEED"] = str(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import shutil
import glob
import numpy as np
import tensorflow as tf

import models
import utils
import dataset

# ----------------------------------------------------------------
# Arguments and Settings

# random seeds
random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# copy python files for reproducibility
logger, dirname = utils.setup_logging(args)
files_to_copy = glob.glob(os.path.dirname(os.path.realpath(__file__)) + "/*.py")
for file_ in files_to_copy:
    script_src = file_
    script_dst = os.path.abspath(os.path.join(dirname, os.path.basename(file_)))
    shutil.copyfile(script_src, script_dst)
logger.debug("Copied {} to {} for reproducibility.".format(", ".join(map(os.path.basename, files_to_copy)), dirname))

# global constants & variables
EMBEDDING_DIM = 18
HIDDEN_SIZE = 18 * 2
ATTENTION_SIZE = 18 * 2
best_auc = 0.0

# print arguments
for k, v in sorted(vars(args).items()):
    logger.info("  %20s: %s" % (k, v))

# get arguments
method = args.method
n_epochs = args.epochs
batch_size = args.batch_size
maxlen = args.maxlen
lr = args.lr
lr_decay = args.lr_decay
momentum = args.momentum
layer_dims = utils.get_ints(args.dims)
temp = args.temp

gp_params = dict()
gp_params["num_inducing"] = args.n_ind
gp_params["lengthscale"] = args.lengthscale
gp_params["amplitude"] = args.amplitude
gp_params["jitter"] = args.jitter
gp_params["n_gh_samples"] = 20
gp_params["n_mc_samples"] = 2000
gp_params["prior_mean"] = args.prior_mean
gp_params["diag_cov"] = args.diag_cov
gp_params["km_coeff"] = args.km_coeff

# ----------------------------------------------------------------
# Dataset

logger.info("Dataset {} loading...".format(args.dataset))

if args.dataset in ["Books", "Electronics"]:
    data_path = "data/" + args.dataset
else:
    logger.error("Invalid dataset : {}".format(args.dataset))
    raise ValueError("Invalid dataset : {}".format(args.dataset))

train_data = dataset.DataIterator("local_train_splitByUser", data_path, batch_size, maxlen)
test_data = dataset.DataIterator("local_test_splitByUser", data_path, batch_size, maxlen)
n_uid, n_mid, n_cat = train_data.get_n()

logger.info("Dataset {} loaded.".format(args.dataset))
logger.info("# UID: {}, # MID: {}, # CAT: {}.".format(n_uid, n_mid, n_cat))


# helper function for converting data to dense vectors
def prepare_data(input, target, maxlen=None, return_neg=False):
    lengths_x = [len(inp[4]) for inp in input]
    seqs_mid = [inp[3] for inp in input]
    seqs_cat = [inp[4] for inp in input]
    noclk_seqs_mid = [inp[5] for inp in input]
    noclk_seqs_cat = [inp[6] for inp in input]

    if maxlen is not None:
        new_seqs_mid = []
        new_seqs_cat = []
        new_noclk_seqs_mid = []
        new_noclk_seqs_cat = []
        new_lengths_x = []
        for l_x, inp in zip(lengths_x, input):
            if l_x > maxlen:
                new_seqs_mid.append(inp[3][l_x - maxlen :])
                new_seqs_cat.append(inp[4][l_x - maxlen :])
                new_noclk_seqs_mid.append(inp[5][l_x - maxlen :])
                new_noclk_seqs_cat.append(inp[6][l_x - maxlen :])
                new_lengths_x.append(maxlen)
            else:
                new_seqs_mid.append(inp[3])
                new_seqs_cat.append(inp[4])
                new_noclk_seqs_mid.append(inp[5])
                new_noclk_seqs_cat.append(inp[6])
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_mid = new_seqs_mid
        seqs_cat = new_seqs_cat
        noclk_seqs_mid = new_noclk_seqs_mid
        noclk_seqs_cat = new_noclk_seqs_cat

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_mid)
    maxlen_x = np.max(lengths_x)
    neg_samples = len(noclk_seqs_mid[0][0])

    mid_his = np.zeros((n_samples, maxlen_x)).astype("int64")
    cat_his = np.zeros((n_samples, maxlen_x)).astype("int64")
    noclk_mid_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype("int64")
    noclk_cat_his = np.zeros((n_samples, maxlen_x, neg_samples)).astype("int64")
    mid_mask = np.zeros((n_samples, maxlen_x)).astype("float32")
    for idx, [s_x, s_y, no_sx, no_sy] in enumerate(zip(seqs_mid, seqs_cat, noclk_seqs_mid, noclk_seqs_cat)):
        mid_mask[idx, : lengths_x[idx]] = 1.0
        mid_his[idx, : lengths_x[idx]] = s_x
        cat_his[idx, : lengths_x[idx]] = s_y
        noclk_mid_his[idx, : lengths_x[idx], :] = no_sx
        noclk_cat_his[idx, : lengths_x[idx], :] = no_sy

    uids = np.array([inp[0] for inp in input])
    mids = np.array([inp[1] for inp in input])
    cats = np.array([inp[2] for inp in input])

    if return_neg:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x), noclk_mid_his, noclk_cat_his
    else:
        return uids, mids, cats, mid_his, cat_his, mid_mask, np.array(target), np.array(lengths_x)


# ----------------------------------------------------------------
# Model setup

# base models
if method == "DNN":
    model = models.Model_DNN(
        n_uid, n_mid, n_cat, EMBEDDING_DIM, layer_dims=layer_dims, optm=args.optimizer, beta1=momentum, gp_params_dict=gp_params
    )
elif method == "PNN":
    model = models.Model_PNN(
        n_uid, n_mid, n_cat, EMBEDDING_DIM, layer_dims=layer_dims, optm=args.optimizer, beta1=momentum, gp_params_dict=gp_params
    )
elif method == "Wide":
    model = models.Model_WideDeep(
        n_uid, n_mid, n_cat, EMBEDDING_DIM, layer_dims=layer_dims, optm=args.optimizer, beta1=momentum, gp_params_dict=gp_params
    )
elif method == "DIN":
    model = models.Model_DIN(
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        ATTENTION_SIZE,
        layer_dims=layer_dims,
        optm=args.optimizer,
        beta1=momentum,
        gp_params_dict=gp_params,
    )
elif method == "DIEN":
    model = models.Model_DIEN(
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        HIDDEN_SIZE,
        ATTENTION_SIZE,
        layer_dims=layer_dims,
        optm=args.optimizer,
        beta1=momentum,
        gp_params_dict=gp_params,
    )
else:
    logger.error("Invalid method : {}".format(method))
    raise ValueError("Invalid method : {}".format(method))

# ----------------------------------------------------------------
# Training

GPU_OPTIONS = tf.GPUOptions(allow_growth=True)
CONFIG = tf.ConfigProto(gpu_options=GPU_OPTIONS)
sess = tf.Session(config=CONFIG)
global_init_op = tf.global_variables_initializer()
sess.run(global_init_op)

writer = tf.summary.FileWriter(dirname + "/summary/")


def evaluate(sess, test_data, model):
    test_loss_sum = 0.0
    test_accuracy_sum = 0.0
    test_aux_loss_sum = 0.0
    nums = 0
    stored_arr = []
    for src, tgt in test_data:
        nums += 1
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(src, tgt, return_neg=True)
        prob, loss, acc, aux_loss = model.calculate(
            sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats, temp]
        )

        test_loss_sum += loss
        test_accuracy_sum += acc
        test_aux_loss_sum = aux_loss
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])

    test_auc = utils.calc_auc(stored_arr)
    test_loss_avg = test_loss_sum / nums
    test_accuracy_avg = test_accuracy_sum / nums
    test_aux_loss_avg = test_aux_loss_sum / nums
    return test_auc, test_loss_avg, test_accuracy_avg, test_aux_loss_avg


# helper function for adding summary (only simple_value supported)
def write_summary(_writer, tag, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    _writer.add_summary(summary, step)
    _writer.flush()


saver = tf.train.Saver(max_to_keep=None)
if args.model_path:
    saver.restore(sess, args.model_path)
    logger.info("Loaded model from {}".format(args.model_path))

# print variables
logger.debug("Model Variables:")
for p in tf.trainable_variables():
    logger.debug("%s: %s" % (p.name, sess.run(tf.shape(p))))

# start training
step = 0
loss_list, acc_list, aux_loss_list = [], [], []
test_auc_list, test_loss_list, test_acc_list, test_aux_loss_list = [], [], [], []
for epoch in range(n_epochs):
    for src, tgt in train_data:
        uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, noclk_mids, noclk_cats = prepare_data(
            src, tgt, maxlen, return_neg=True
        )
        loss, acc, aux_loss, smr = model.train(
            sess, [uids, mids, cats, mid_his, cat_his, mid_mask, target, sl, lr, noclk_mids, noclk_cats, temp]
        )

        step += 1
        loss_list.append(loss)
        acc_list.append(acc)
        aux_loss_list.append(aux_loss)

        # print training metrics
        if step % args.log_every == 0:
            logger.info(
                "step: {:11d}: train_loss = {:.5f}, train_accuracy = {:.5f}, train_aux_loss = {:.5f}".format(
                    step,
                    np.mean(loss_list[-args.log_every :]),
                    np.mean(acc_list[-args.log_every :]),
                    np.mean(aux_loss_list[-args.log_every :]),
                )
            )
            writer.add_summary(smr, step)
            writer.flush()

        # test and visualization
        if step % args.viz_every == 0:
            test_auc, test_loss, test_accuracy, test_aux_loss = evaluate(sess, test_data, model)
            logger.critical(
                "test_auc: {:.5f}:  test_loss = {:.5f},  test_accuracy = {:.5f},  test_aux_loss = {:.5f}".format(
                    test_auc, test_loss, test_accuracy, test_aux_loss
                )
            )
            test_auc_list.append(test_auc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_accuracy)
            test_aux_loss_list.append(test_aux_loss)
            write_summary(writer, tag="Test/auc", value=test_auc, step=step)
            write_summary(writer, tag="Test/accuracy", value=test_accuracy, step=step)
            write_summary(writer, tag="Test/loss", value=test_loss, step=step)

            if best_auc < test_auc:
                best_auc = test_auc
                saver.save(sess, dirname + "/best_model/")
                logger.warning("[{}] Saved best model at step: {}, auc = {:.5f}".format(args.message, step, best_auc))

        # save model
        if step % args.model_every == 0:
            saver.save(sess, dirname + "/model/", global_step=step)
            logger.info("Saved model at step: {}".format(step))

    # learning rate decay after each epoch
    logger.debug("Epoch {:3d} finished. Learning rate reduced from {:.4E} to {:.4E}".format(epoch + 1, lr, lr * lr_decay))
    lr *= lr_decay

logger.error("Experiment [{}] finished. Best auc = {:.5f}.".format(args.message, best_auc))
