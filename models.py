from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell
from patches.rnn import dynamic_rnn
from patches.attention import *
from patches.activation import *


class Model(object):
    def __init__(
        self,
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        layer_dims=[200, 80],
        use_negsampling=False,
        optm="Adam",
        beta1=0.9,
        gp_params_dict=None,
    ):
        # Placeholders
        with tf.name_scope("inputs"):
            self.uid_batch_ph = tf.placeholder(tf.int32, [None], name="uid_batch_ph")
            self.mid_batch_ph = tf.placeholder(tf.int32, [None], name="mid_batch_ph")
            self.target_ph = tf.placeholder(tf.float32, [None, None], name="target_ph")
            self.lr = tf.placeholder(tf.float32, [], name="lr")
            self.temp = tf.placeholder(tf.float32, [], name="temp")
            self.mid_his_batch_ph = tf.placeholder(tf.int32, [None, None], name="mid_his_batch_ph")
            self.cat_his_batch_ph = tf.placeholder(tf.int32, [None, None], name="cat_his_batch_ph")
            self.cat_batch_ph = tf.placeholder(tf.int32, [None], name="cat_batch_ph")
            self.mask = tf.placeholder(tf.float32, [None, None], name="mask")
            self.seq_len_ph = tf.placeholder(tf.int32, [None], name="seq_len_ph")
            self.use_negsampling = use_negsampling
            if use_negsampling:
                self.noclk_mid_batch_ph = tf.placeholder(tf.int32, [None, None, None], name="noclk_mid_batch_ph")
                self.noclk_cat_batch_ph = tf.placeholder(tf.int32, [None, None, None], name="noclk_cat_batch_ph")

        # Embedding layers
        with tf.variable_scope("embedding"):
            self.uid_embeddings_var = tf.get_variable("uid_embedding_var", [n_uid, EMBEDDING_DIM])
            tf.summary.histogram("uid_embeddings_var", self.uid_embeddings_var)
            self.uid_batch_embedded = tf.nn.embedding_lookup(self.uid_embeddings_var, self.uid_batch_ph)

            self.mid_embeddings_var = tf.get_variable("mid_embedding_var", [n_mid, EMBEDDING_DIM])
            tf.summary.histogram("mid_embeddings_var", self.mid_embeddings_var)
            self.mid_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_batch_ph)
            self.mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.mid_his_batch_ph)
            if self.use_negsampling:
                self.noclk_mid_his_batch_embedded = tf.nn.embedding_lookup(self.mid_embeddings_var, self.noclk_mid_batch_ph)

            self.cat_embeddings_var = tf.get_variable("cat_embedding_var", [n_cat, EMBEDDING_DIM])
            tf.summary.histogram("cat_embeddings_var", self.cat_embeddings_var)
            self.cat_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_batch_ph)
            self.cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.cat_his_batch_ph)
            if self.use_negsampling:
                self.noclk_cat_his_batch_embedded = tf.nn.embedding_lookup(self.cat_embeddings_var, self.noclk_cat_batch_ph)

        self.item_eb = tf.concat([self.mid_batch_embedded, self.cat_batch_embedded], 1)
        self.item_his_eb = tf.concat([self.mid_his_batch_embedded, self.cat_his_batch_embedded], 2)
        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)
        if self.use_negsampling:
            self.noclk_item_his_eb = tf.concat(
                [self.noclk_mid_his_batch_embedded[:, :, 0, :], self.noclk_cat_his_batch_embedded[:, :, 0, :]], -1
            )
            self.noclk_item_his_eb = tf.reshape(
                self.noclk_item_his_eb, [-1, tf.shape(self.noclk_mid_his_batch_embedded)[1], EMBEDDING_DIM * 2]
            )

            self.noclk_his_eb = tf.concat([self.noclk_mid_his_batch_embedded, self.noclk_cat_his_batch_embedded], -1)
            self.noclk_his_eb_sum_1 = tf.reduce_sum(self.noclk_his_eb, 2)
            self.noclk_his_eb_sum = tf.reduce_sum(self.noclk_his_eb_sum_1, 1)

        # Settings
        self.layer_dims = layer_dims
        self.beta1 = beta1
        self.epsilon = 1e-08
        self.optm = optm
        assert optm.lower() in ["adam", "momentum"]
        self.with_wide = False
        self.gp_params = gp_params_dict

    def build_network(self, inp, activation="prelu"):
        layer_dims = self.layer_dims
        with tf.variable_scope("network"):
            hid = tf.layers.batch_normalization(inputs=inp, name="bn_0")
            for i in range(len(layer_dims)):
                hid = tf.layers.dense(hid, layer_dims[i], activation=None, name="layer_" + str(i + 1))
                if activation == "relu":
                    hid = tf.nn.relu(hid)
                elif activation == "tanh":
                    hid = tf.nn.tanh(hid)
                elif activation == "sigmoid":
                    hid = tf.nn.sigmoid(hid)
                elif activation == "prelu":
                    hid = prelu(hid, "prelu_" + str(i + 1))
                elif activation == "dice":
                    hid = dice(hid, name="dice_" + str(i + 1))
                else:
                    raise ValueError("Wrong activation function")
            self.hid = hid

    def build_GP_elbo_loss(self, inp, activation="prelu"):
        # GP hyper-parameters
        gp_params = self.gp_params
        num_inducing = gp_params["num_inducing"] if "num_inducing" in gp_params else 200
        lengthscale = gp_params["lengthscale"] if "lengthscale" in gp_params else 2.0
        amplitude = gp_params["amplitude"] if "amplitude" in gp_params else 0.3
        jitter = gp_params["jitter"] if "jitter" in gp_params else 1e-4
        n_gh_samples = gp_params["n_gh_samples"] if "n_gh_samples" in gp_params else 20
        n_mc_samples = gp_params["n_mc_samples"] if "n_mc_samples" in gp_params else 2000
        prior_mean = gp_params["prior_mean"] if "prior_mean" in gp_params else 0.0
        diag_cov = gp_params["diag_cov"] if "diag_cov" in gp_params else False
        D = gp_params["hidden_dims"] if "hidden_dims" in gp_params else 2
        km_coeff = gp_params["km_coeff"] if "km_coeff" in gp_params else 0.0

        # Kernel function
        def broadcasting_elementwise(op, a, b):
            flatres = op(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
            return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))

        def square_distance(X1, X2):
            if X2 is None:
                X1s = tf.reduce_sum(tf.square(X1), axis=-1, keep_dims=True)
                dist = -2 * tf.matmul(X1, X1, transpose_b=True)
                dist += X1s + tf.linalg.adjoint(X1s)
                return dist
            X1s = tf.reduce_sum(tf.square(X1), axis=-1)
            X2s = tf.reduce_sum(tf.square(X2), axis=-1)
            dist = -2 * tf.tensordot(X1, X2, [[-1], [-1]])
            dist += broadcasting_elementwise(tf.add, X1s, X2s)
            return dist

        def kernel(X1, X2):
            square_dist = square_distance(X1 / lengthscale, X2 / lengthscale)
            return amplitude * amplitude * tf.exp(-0.5 * square_dist)

        def inv_link_func(x_):
            return tf.sigmoid(x_)

        self.build_network(inp, activation=activation)

        if self.layer_dims:
            with tf.variable_scope("adapter"):
                self.hid = tf.layers.dense(self.hid, D, activation=None, name="layer_adapter")
                if self.with_wide:
                    d_layer_wide = tf.concat(
                        [tf.concat([self.item_eb, self.item_his_eb_sum], axis=-1), self.item_eb * self.item_his_eb_sum], axis=-1
                    )
                    d_layer_wide = tf.layers.dense(d_layer_wide, D, activation=None, name="f_fm")
                    self.hid = self.hid + d_layer_wide

        X = self.hid
        Y = self.target_ph

        # Variational parameters
        with tf.variable_scope("variational"):
            Z = tf.get_variable("ind_point", shape=(num_inducing, D), dtype=tf.float32)
            q_mu = tf.get_variable(
                "ind_mean", shape=(num_inducing, 1), initializer=tf.initializers.constant(prior_mean), dtype=tf.float32
            )
            if diag_cov:
                q_cov_variable = tf.get_variable(
                    "ind_cov_flat", shape=[num_inducing], initializer=tf.initializers.constant(amplitude), dtype=tf.float32
                )
            else:
                q_cov_variable = tf.get_variable(
                    "ind_cov_flat",
                    shape=((num_inducing + 1) * num_inducing / 2,),
                    initializer=tf.initializers.constant(
                        amplitude * np.eye(num_inducing, dtype=np.float32)[np.tril_indices(num_inducing)]
                    ),
                    dtype=tf.float32,
                )
        self.Z = Z
        self.q_mu = q_mu

        if diag_cov:
            Lq = tf.linalg.tensor_diag(q_cov_variable)
        else:
            indices = list(zip(*np.tril_indices(num_inducing)))
            indices = tf.constant([list(i) for i in indices], dtype=tf.int32)
            Lq = tf.scatter_nd(indices=indices, shape=(num_inducing, num_inducing), updates=q_cov_variable)
        q_cov = tf.matmul(Lq, Lq, transpose_b=True)
        Lq_diag = tf.linalg.diag_part(Lq)
        self.q_cov = q_cov

        # KL divergence
        KZZ = kernel(Z, Z) + jitter * tf.eye(num_inducing, dtype=tf.float32)
        self.KZZ = KZZ
        Lp = tf.linalg.cholesky(KZZ)
        p_mu = prior_mean

        quadratic = tf.linalg.triangular_solve(Lp, q_mu - p_mu, lower=True)
        mahalanobis = tf.reduce_sum(tf.square(quadratic))

        Lp_inv_Lq = tf.linalg.triangular_solve(Lp, Lq, lower=True)
        trace = tf.reduce_sum(tf.square(Lp_inv_Lq))

        logdet_qcov = tf.reduce_sum(tf.log(self.epsilon + tf.square(Lq_diag)))
        logdet_pcov = tf.reduce_sum(tf.log(self.epsilon + tf.square(tf.linalg.diag_part(Lp))))

        constant = -num_inducing

        KL_divergence = 0.5 * (logdet_pcov - logdet_qcov + trace + mahalanobis + constant)

        # Mean and Variance
        KZX = kernel(Z, X)
        Lp_inv_KZX = tf.linalg.triangular_solve(Lp, KZX, lower=True)
        KZZ_inv_KZX = tf.linalg.triangular_solve(tf.transpose(Lp), Lp_inv_KZX, lower=False)
        f_mean = p_mu + tf.matmul(KZZ_inv_KZX, (q_mu - p_mu), transpose_a=True)
        self.logit = tf.squeeze(f_mean)

        KXX_diag = amplitude * amplitude * tf.ones([tf.shape(X)[0], 1])
        KXZ_KZZ_inv_Lq = tf.matmul(KZZ_inv_KZX, Lq, transpose_a=True)
        f_var_1 = tf.reduce_sum(tf.square(tf.transpose(Lp_inv_KZX)), axis=1, keep_dims=True)
        f_var_2 = tf.reduce_sum(tf.square(KXZ_KZZ_inv_Lq), axis=1, keep_dims=True)
        f_var = KXX_diag - f_var_1 + f_var_2
        self.logit_var = tf.squeeze(f_var)

        KXX = kernel(X, X) + jitter * tf.eye(tf.shape(X)[0], dtype=tf.float32)
        f_var_full_1 = tf.matmul(Lp_inv_KZX, Lp_inv_KZX, transpose_a=True)
        f_var_full_2 = tf.matmul(KXZ_KZZ_inv_Lq, KXZ_KZZ_inv_Lq, transpose_b=True)
        f_var_full = KXX - f_var_full_1 + f_var_full_2

        # Likelihood (Gauss-Hermite)
        x_gh, w_gh = np.polynomial.hermite.hermgauss(n_gh_samples)
        x_gh, w_gh = x_gh.astype(np.float32).reshape(1, -1), w_gh.astype(np.float32).reshape(1, -1, 1)
        w_gh = w_gh / np.sqrt(np.pi)

        x_logit_samples = x_gh * tf.sqrt(2.0 * f_var) + f_mean
        pctr_samples = tf.expand_dims(inv_link_func(x_logit_samples), axis=2)
        pctr_samples_with_noclk = tf.concat([pctr_samples, 1 - pctr_samples], axis=2)
        variational_ep_ll = tf.reduce_sum(
            w_gh * tf.reshape(Y, shape=(-1, 1, 2)) * tf.log(pctr_samples_with_noclk + self.epsilon), axis=(1, 2)
        )
        Likelihood = tf.reduce_mean(variational_ep_ll)

        # hidden summaries
        with tf.name_scope("Hidden"):
            tf.summary.histogram("hidden", self.hid)
            tf.summary.histogram("ind_point", self.Z)
            tf.summary.histogram("ind_mean", self.q_mu)
            tf.summary.histogram("logit", self.logit)
            tf.summary.histogram("logit_var", self.logit_var)

        # Regularization
        # ZX_sq_dist = square_distance(Z, tf.stop_gradient(X))
        ZX_sq_dist = square_distance(Z, X)
        assignments_X = tf.argmin(ZX_sq_dist, axis=0, output_type=tf.int32)
        assignments_Z = tf.argmin(ZX_sq_dist, axis=1, output_type=tf.int32)
        batch_idx = tf.range(tf.shape(assignments_X)[0], dtype=tf.int32)
        centroids_idx = tf.range(tf.shape(assignments_Z)[0], dtype=tf.int32)
        batch_nearest = tf.concat([tf.expand_dims(assignments_X, 1), tf.expand_dims(batch_idx, 1)], axis=1)
        centroids_nearest = tf.concat([tf.expand_dims(centroids_idx, 1), tf.expand_dims(assignments_Z, 1)], axis=1)
        dist_2_nearest_centroid = tf.gather_nd(ZX_sq_dist, batch_nearest)
        dist_2_nearest_sample = tf.gather_nd(ZX_sq_dist, centroids_nearest)
        kmeans_loss_1 = tf.reduce_mean(dist_2_nearest_centroid)
        kmeans_loss_2 = tf.reduce_mean(dist_2_nearest_sample)
        kmeans_loss = kmeans_loss_1 + kmeans_loss_2

        # ELBO (with regularization)
        loss = -Likelihood + self.temp * KL_divergence + km_coeff * kmeans_loss
        self.loss = loss
        if self.use_negsampling:
            self.loss += self.aux_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.optm.lower() == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=self.beta1, epsilon=self.epsilon).minimize(
                    self.loss
                )
            elif self.optm.lower() == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=self.beta1).minimize(self.loss)

        # Others
        pctr_naive = inv_link_func(f_mean)
        prop_naive = tf.concat([pctr_naive, 1 - pctr_naive], axis=1)
        self.y_hat = prop_naive

        prop_gh = tf.reduce_sum(w_gh * pctr_samples_with_noclk, axis=1)

        x_logit_samples_mc = tf.random_normal([tf.shape(f_mean)[0], n_mc_samples], dtype=tf.float32) * tf.sqrt(f_var) + f_mean
        prop_mc, prop_mc_var = tf.nn.moments(inv_link_func(x_logit_samples_mc), axes=[1], keep_dims=True)
        prop_mc = tf.concat([prop_mc, 1 - prop_mc], axis=1)

        NLL_lg = -tf.reduce_mean(tf.reduce_sum(tf.log(prop_naive) * Y, axis=1))
        NLL_gh = -tf.reduce_mean(tf.reduce_sum(tf.log(prop_gh) * Y, axis=1))
        NLL_mc = -tf.reduce_mean(tf.reduce_sum(tf.log(prop_mc) * Y, axis=1))

        # Summary
        with tf.name_scope("Metrics"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("lr", self.lr)

            # Accuracy metric
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

            # GP metrics
            tf.summary.scalar("Likelihood", Likelihood)
            tf.summary.scalar("KL", KL_divergence)
            tf.summary.scalar("NLL_lg", NLL_lg)
            tf.summary.scalar("NLL_gh", NLL_gh)
            tf.summary.scalar("NLL_mc", NLL_mc)
            tf.summary.scalar("kmeans_loss", kmeans_loss)

        self.merged = tf.summary.merge_all()

    def train(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cat_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cat_his_batch_ph: inps[4],
            self.mask: inps[5],
            self.target_ph: inps[6],
            self.seq_len_ph: inps[7],
            self.lr: inps[8],
            self.temp: inps[11],
        }
        if self.use_negsampling:
            feed_dict[self.noclk_mid_batch_ph] = inps[9]
            feed_dict[self.noclk_cat_batch_ph] = inps[10]
            loss, accuracy, aux_loss, _, summary = sess.run(
                [self.loss, self.accuracy, self.aux_loss, self.optimizer, self.merged], feed_dict=feed_dict
            )
            return loss, accuracy, aux_loss, summary
        else:
            loss, accuracy, _, summary = sess.run([self.loss, self.accuracy, self.optimizer, self.merged], feed_dict=feed_dict)
            return loss, accuracy, 0, summary

    def calculate(self, sess, inps):
        feed_dict = {
            self.uid_batch_ph: inps[0],
            self.mid_batch_ph: inps[1],
            self.cat_batch_ph: inps[2],
            self.mid_his_batch_ph: inps[3],
            self.cat_his_batch_ph: inps[4],
            self.mask: inps[5],
            self.target_ph: inps[6],
            self.seq_len_ph: inps[7],
            self.temp: inps[10],
        }
        if self.use_negsampling:
            feed_dict[self.noclk_mid_batch_ph] = inps[8]
            feed_dict[self.noclk_cat_batch_ph] = inps[9]
            probs, loss, accuracy, aux_loss = sess.run([self.y_hat, self.loss, self.accuracy, self.aux_loss], feed_dict=feed_dict)
            return probs, loss, accuracy, aux_loss
        else:
            probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict=feed_dict)
            return probs, loss, accuracy, 0


class Model_DNN(Model):
    def __init__(
        self,
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        layer_dims=[200, 80],
        use_negsampling=False,
        optm="Adam",
        beta1=0.9,
        gp_params_dict=None,
    ):
        super(Model_DNN, self).__init__(
            n_uid,
            n_mid,
            n_cat,
            EMBEDDING_DIM,
            layer_dims=layer_dims,
            use_negsampling=use_negsampling,
            optm=optm,
            beta1=beta1,
            gp_params_dict=gp_params_dict,
        )

        self.inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        self.build_GP_elbo_loss(self.inp, activation="prelu")


class Model_PNN(Model):
    def __init__(
        self,
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        layer_dims=[200, 80],
        use_negsampling=False,
        optm="Adam",
        beta1=0.9,
        gp_params_dict=None,
    ):
        super(Model_PNN, self).__init__(
            n_uid,
            n_mid,
            n_cat,
            EMBEDDING_DIM,
            layer_dims=layer_dims,
            use_negsampling=use_negsampling,
            optm=optm,
            beta1=beta1,
            gp_params_dict=gp_params_dict,
        )

        self.inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum], 1
        )
        self.build_GP_elbo_loss(self.inp, activation="prelu")


class Model_WideDeep(Model):
    def __init__(
        self,
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        layer_dims=[200, 80],
        use_negsampling=False,
        optm="Adam",
        beta1=0.9,
        gp_params_dict=None,
    ):
        super(Model_WideDeep, self).__init__(
            n_uid,
            n_mid,
            n_cat,
            EMBEDDING_DIM,
            layer_dims=layer_dims,
            use_negsampling=use_negsampling,
            optm=optm,
            beta1=beta1,
            gp_params_dict=gp_params_dict,
        )

        self.with_wide = True
        self.inp = tf.concat([self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum], 1)
        self.build_GP_elbo_loss(self.inp, activation="prelu")


class Model_DIN(Model):
    def __init__(
        self,
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        ATTENTION_SIZE,
        layer_dims=[200, 80],
        use_negsampling=False,
        optm="Adam",
        beta1=0.9,
        gp_params_dict=None,
    ):
        super(Model_DIN, self).__init__(
            n_uid,
            n_mid,
            n_cat,
            EMBEDDING_DIM,
            layer_dims=layer_dims,
            use_negsampling=use_negsampling,
            optm=optm,
            beta1=beta1,
            gp_params_dict=gp_params_dict,
        )

        with tf.variable_scope("network"):
            with tf.variable_scope("Attention_layer"):
                attention_output = din_attention(self.item_eb, self.item_his_eb, ATTENTION_SIZE, self.mask)
                att_fea = tf.reduce_sum(attention_output, 1)
                tf.summary.histogram("att_fea", att_fea)

        self.inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, att_fea], -1
        )
        self.build_GP_elbo_loss(self.inp, activation="dice")


class Model_DIEN(Model):
    def __init__(
        self,
        n_uid,
        n_mid,
        n_cat,
        EMBEDDING_DIM,
        HIDDEN_SIZE,
        ATTENTION_SIZE,
        layer_dims=[200, 80],
        use_negsampling=True,
        optm="Adam",
        beta1=0.9,
        gp_params_dict=None,
    ):
        super(Model_DIEN, self).__init__(
            n_uid,
            n_mid,
            n_cat,
            EMBEDDING_DIM,
            layer_dims=layer_dims,
            use_negsampling=use_negsampling,
            optm=optm,
            beta1=beta1,
            gp_params_dict=gp_params_dict,
        )

        with tf.variable_scope("network"):
            with tf.variable_scope("rnn_1"):
                rnn_outputs, _ = dynamic_rnn(
                    GRUCell(HIDDEN_SIZE),
                    inputs=self.item_his_eb,
                    sequence_length=self.seq_len_ph,
                    dtype=tf.float32,
                    scope="gru_1",
                )
                tf.summary.histogram("GRU_outputs", rnn_outputs)

            aux_loss_1 = self.auxiliary_loss(
                rnn_outputs[:, :-1, :],
                self.item_his_eb[:, 1:, :],
                self.noclk_item_his_eb[:, 1:, :],
                self.mask[:, 1:],
                stag="_aux",
            )
            self.aux_loss = aux_loss_1

            with tf.variable_scope("attn_1"):
                _, alphas = din_fcn_attention(
                    self.item_eb,
                    rnn_outputs,
                    ATTENTION_SIZE,
                    self.mask,
                    softmax_stag=1,
                    stag="1_1",
                    mode="LIST",
                    return_alphas=True,
                )
                tf.summary.histogram("alpha_outputs", alphas)

            with tf.variable_scope("rnn_2"):
                _, final_state2 = dynamic_rnn(
                    VecAttGRUCell(HIDDEN_SIZE),
                    inputs=rnn_outputs,
                    att_scores=tf.expand_dims(alphas, -1),
                    sequence_length=self.seq_len_ph,
                    dtype=tf.float32,
                    scope="gru_2",
                )
                tf.summary.histogram("GRU2_Final_State", final_state2)

        self.inp = tf.concat(
            [self.uid_batch_embedded, self.item_eb, self.item_his_eb_sum, self.item_eb * self.item_his_eb_sum, final_state2], 1
        )
        self.build_GP_elbo_loss(self.inp, activation="dice")

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask, stag=None):
        mask = tf.cast(mask, tf.float32)
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]
        click_loss_ = -tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = -tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask
        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def auxiliary_net(self, in_, stag="auxiliary_net"):
        bn1 = tf.layers.batch_normalization(inputs=in_, name="bn_0" + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name="layer_1" + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name="layer_2" + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name="layer_3" + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + self.epsilon
        return y_hat
