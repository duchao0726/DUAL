import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from activation import prelu


class VecAttGRUCell(RNNCell):
    def __init__(self, num_units, activation=None, reuse=None, kernel_initializer=None, bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer,
                )

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer,
                )
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def din_attention(
    query, facts, attention_size, mask, stag="null", mode="SUM", softmax_stag=1, time_major=False, return_alphas=False
):
    if isinstance(facts, tuple):
        facts = tf.concat(facts, 2)
        print("querry_size mismatch")
        query = tf.concat(values=[query, query], axis=1)

    if time_major:
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name="f1_att" + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name="f2_att" + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name="f3_att" + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    key_masks = tf.expand_dims(mask, 1)
    paddings = tf.ones_like(scores) * (-(2 ** 32) + 1)
    scores = tf.where(key_masks, scores, paddings)

    if softmax_stag:
        scores = tf.nn.softmax(scores)

    if mode == "SUM":
        output = tf.matmul(scores, facts)
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output


def din_fcn_attention(
    query,
    facts,
    attention_size,
    mask,
    stag="null",
    mode="SUM",
    softmax_stag=1,
    time_major=False,
    return_alphas=False,
    forCnn=False,
):
    if isinstance(facts, tuple):
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name="f1" + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name="f1_att" + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name="f2_att" + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name="f3_att" + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    key_masks = tf.expand_dims(mask, 1)
    paddings = tf.ones_like(scores) * (-(2 ** 32) + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)

    if softmax_stag:
        scores = tf.nn.softmax(scores)

    if mode == "SUM":
        output = tf.matmul(scores, facts)
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output
