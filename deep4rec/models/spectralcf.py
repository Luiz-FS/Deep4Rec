import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
from tqdm import tqdm

from deep4rec.models.model import Model


class SpectralCF(Model):
    def __init__(self, ds, K=3, emb_dim=16, lr=0.001, batch_size=1024, decay=0.001):
        super(SpectralCF, self).__init__()

        self.graph = ds.build_graph()
        self.num_items = ds.num_items
        self.num_users = ds.num_users
        self.K = K
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.decay = decay
        self.ds = ds
        self.lr = lr

        self.matrix_adj = self.adjacent_matrix()
        self.matrix_d = self.degree_matrix()
        self.matrix_l = self.laplacian_matrix(normalized=True)

        self.lamda, self.U = np.linalg.eig(self.matrix_l)
        self.lamda = np.diag(self.lamda)

        self.users = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.pos_items = tf.placeholder(tf.int32, shape=(self.batch_size,))
        self.neg_items = tf.placeholder(tf.int32, shape=(self.batch_size,))

        self.user_embeddings = tfe.Variable(
            tf.random_normal(
                [self.num_users, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32
            ),
            name="user_embeddings",
        )
        self.item_embeddings = tfe.Variable(
            tf.random_normal(
                [self.num_items, emb_dim], mean=0.01, stddev=0.02, dtype=tf.float32
            ),
            name="item_embeddings",
        )

        A_hat = np.dot(self.U, self.U.T) + np.dot(np.dot(self.U, self.lamda), self.U.T)
        A_hat = A_hat.astype(np.float32)

        self.filters = []
        for k in range(self.K):
            self.filters.append(
                tfe.Variable(
                    tf.random_normal(
                        [self.emb_dim, self.emb_dim],
                        mean=0.01,
                        stddev=0.02,
                        dtype=tf.float32,
                    )
                )
            )

        embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.K):
            embeddings = tf.matmul(A_hat, embeddings)
            embeddings = tf.nn.sigmoid(tf.matmul(embeddings, self.filters[k]))
            all_embeddings += [embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        self.u_embeddings, self.i_embeddings = tf.split(
            all_embeddings, [self.num_users, self.num_items], 0
        )

        self.u_embeddings = tf.nn.embedding_lookup(self.u_embeddings, self.users)
        self.pos_i_embeddings = tf.nn.embedding_lookup(
            self.i_embeddings, self.pos_items
        )
        self.neg_i_embeddings = tf.nn.embedding_lookup(
            self.i_embeddings, self.neg_items
        )

        self.loss = self.create_bpr_loss(
            self.u_embeddings, self.pos_i_embeddings, self.neg_i_embeddings
        )
        self.opt = tf.train.RMSPropOptimizer(learning_rate=lr)
        self.updatess = self.opt.minimize(
            self.loss,
            var_list=[self.user_embeddings, self.item_embeddings] + self.filters,
        )

    def adjacent_matrix(self, self_connection=False):
        matrix_adj = np.zeros(
            [self.num_users + self.num_items, self.num_users + self.num_items],
            dtype=np.float32,
        )

        matrix_adj[: self.num_users, self.num_users :] = self.graph
        matrix_adj[self.num_users :, : self.num_users] = self.graph.T

        if self_connection:
            return (
                np.identity(self.n_users + self.n_items, dtype=np.float32) + matrix_adj
            )

        return matrix_adj

    def degree_matrix(self):
        matrix_d = np.sum(self.matrix_adj, axis=1, keepdims=False)
        matrix_d[matrix_d == 0] = 1e-8
        return matrix_d

    def laplacian_matrix(self, normalized=False):
        if not normalized:
            return self.D - self.A

        tmp = np.dot(np.diag(np.power(self.matrix_d, -1)), self.matrix_adj)
        return np.identity(self.num_users + self.num_items, dtype=np.float32) - tmp

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)

        regularizer = (
            tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        )
        regularizer = regularizer / self.batch_size

        maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        loss = tf.negative(tf.reduce_mean(maxi)) + self.decay * regularizer
        return loss

    def train(
        self,
        ds,
        epochs,
        loss_function,
        batch_size=128,
        optimizer="adam",
        run_eval=True,
        verbose=True,
        eval_metrics=None,
        eval_loss_functions=None,
        train_indexes=None,
        valid_indexes=None,
        early_stop=False,
    ):
        self._losses = {"train": [], "validation": [], "test": []}
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        for epoch in tqdm(range(epochs)):
            users, pos_items, neg_items = ds.sample_pos_neg_items(batch_size)

            (_, loss) = sess.run(
                [self.updatess, self.loss],
                feed_dict={
                    self.users: users,
                    self.pos_items: pos_items,
                    self.neg_items: neg_items,
                },
            )

            self._eval_and_store_results(sess, "train", ds, verbose)
            self._eval_and_store_results(sess, "test", ds, verbose)

    def _eval_and_store_results(self, session, ds_key, ds, verbose):
        loss = self.eval(
            session,
            ds
        )

        if loss:
            self._losses[ds_key].append(loss)
            if verbose:
                self._print_res("%s losses" % ds_key, loss)

    
    def eval(self, session, ds, loss_functions=[], metrics=None):
        users, pos_items, neg_items = ds.sample_pos_neg_items(self.batch_size)

        loss = session.run(
            self.loss,
            feed_dict={
                self.users: users,
                self.pos_items: pos_items,
                self.neg_items: neg_items,
            }
        )

        loss_function_res = {
            "bpr": float(loss)
        }
        
        return loss_function_res

    def call(self, one_hot_features, training=False, features=None, **kwargs):
        features = [[], []]

        for feature in one_hot_features:
            features[0].append(feature[0])
            features[1].append(feature[1])

        users, items = features
        users = list(map(lambda user: self.ds.index_user_id[user.numpy()], users))
        items = list(map(lambda item: self.ds.index_item_id[item.numpy()], items))

        self.u_embeddings = tf.nn.embedding_lookup(self.u_embeddings, users)
        all_ratings = tf.matmul(
            self.u_embeddings, self.i_embeddings, transpose_a=False, transpose_b=True
        )

        users_items = zip(users, items)
        ratings = tf.constant(
            np.array([[all_ratings[user][item]] for (user, item) in users_items])
        )

        return ratings
