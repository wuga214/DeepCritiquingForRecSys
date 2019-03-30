import tensorflow as tf
from tqdm import tqdm
import scipy.sparse as sparse
from utils.reformat import to_sparse_matrix, to_laplacian, to_svd
from providers.sampler import get_negative_sample, get_arrays, concate_data

class CVNCF(object):
    def __init__(self,
                 num_users,
                 num_items,
                 text_dim,
                 embed_dim,
                 num_layers,
                 batch_size,
                 lamb=0.01,
                 learning_rate=1e-4,
                 optimizer=tf.train.RMSPropOptimizer,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        tf.summary.FileWriter('./graphs', self.sess.graph)

    def get_graph(self):

        self.users_index = tf.placeholder(tf.int32, [None], name='user_id')
        self.items_index = tf.placeholder(tf.int32, [None], name='item_id')
        self.rating = tf.placeholder(tf.int32, [None], name='rating')
        self.keyphrase = tf.placeholder(tf.int32, [None, self.text_dim], name='key_phrases')
        self.modified_phrase = tf.placeholder(tf.float32, [None, self.text_dim], name='modified_phrases')
        self.sampling = tf.placeholder(tf.bool)
        self.corruption = tf.placeholder(tf.float32)

        with tf.variable_scope("embeddings"):
            self.user_embeddings = tf.Variable(tf.random_normal([self.num_users, self.embed_dim],
                                                                stddev=1 / (self.embed_dim ** 0.5),
                                                                dtype=tf.float32), trainable=False)

            self.item_embeddings = tf.Variable(tf.random_normal([self.num_items, self.embed_dim],
                                                                stddev=1 / (self.embed_dim ** 0.5),
                                                                dtype=tf.float32), trainable=False)

            users = tf.nn.embedding_lookup(self.user_embeddings, self.users_index, name="user_lookup")
            items = tf.nn.embedding_lookup(self.item_embeddings, self.items_index, name="item_lookup")

        with tf.variable_scope("residual"):
            hi = tf.concat([users, items], axis=1)

            hi = tf.nn.dropout(hi, 1 - self.corruption)

            for i in range(self.num_layers):
                ho = tf.layers.dense(inputs=hi, units=self.embed_dim*2,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                     activation=tf.nn.tanh)
                hi = ho

        with tf.variable_scope('latent'):
            self.mean = hi[:, :self.embed_dim]
            logstd = hi[:, self.embed_dim:]
            self.stddev = tf.exp(logstd)
            epsilon = tf.random_normal(tf.shape(self.stddev))
            self.z = tf.cond(self.sampling, lambda: self.mean + self.stddev * epsilon, lambda: self.mean)

            latent = tf.stop_gradient(hi)

        with tf.variable_scope("prediction", reuse=False):
            rating_prediction = tf.layers.dense(inputs=self.z, units=1,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                                activation=None, name='rating_prediction')
            phrase_prediction = tf.layers.dense(inputs=self.z, units=self.text_dim,
                                                activation=None, name='phrase_prediction')

            self.rating_prediction = rating_prediction
            self.phrase_prediction = phrase_prediction

        with tf.variable_scope("looping"):
            reconstructed_latent = tf.layers.dense(inputs=self.phrase_prediction, units=self.embed_dim*2,
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                                   activation=None, name='latent_reconstruction', reuse=False)

            modified_latent = tf.layers.dense(inputs=self.modified_phrase, units=self.embed_dim*2,
                                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                                   activation=None, name='latent_reconstruction', reuse=True)

            modified_latent = (latent + modified_latent)/2.0
            modified_mean = modified_latent[:, :self.embed_dim]


        with tf.variable_scope("prediction", reuse=True):
            rating_prediction = tf.layers.dense(inputs=modified_mean, units=1,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                                activation=None, name='rating_prediction')
            phrase_prediction = tf.layers.dense(inputs=modified_mean, units=self.text_dim,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                                activation=None, name='phrase_prediction')

            self.modified_rating_prediction = rating_prediction
            self.modified_phrase_prediction = phrase_prediction
        #
        with tf.variable_scope("losses"):

            with tf.variable_scope('kl-divergence'):
                kl = self._kl_diagnormal_stdnormal(self.mean, logstd)

            with tf.variable_scope("latent_reconstruction_loss"):
                latent_loss = tf.losses.mean_squared_error(labels=latent, predictions=reconstructed_latent)

            with tf.variable_scope("rating_loss"):
                # rating_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.reshape(self.rating, [-1, 1]),
                #                                               logits=self.rating_prediction)
                rating_loss = tf.losses.mean_squared_error(labels=tf.reshape(self.rating, [-1, 1]),
                                                           predictions=self.rating_prediction)

            with tf.variable_scope("phrase_loss"):
                phrase_loss = tf.losses.mean_squared_error(labels=self.keyphrase,
                                                           predictions=self.phrase_prediction)

            with tf.variable_scope("l2"):
                l2_loss = tf.losses.get_regularization_loss()

            self.loss = (tf.reduce_mean(rating_loss)
                         + 0.1 * tf.reduce_mean(phrase_loss)
                         + 0.1 * tf.reduce_mean(latent_loss)
                         + kl
                         + l2_loss
                         )

        with tf.variable_scope('optimizer'):
            self.train = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)

    @staticmethod
    def _kl_diagnormal_stdnormal(mu, log_std):
        var_square = tf.exp(2 * log_std)
        kl = 0.5 * tf.reduce_mean(tf.square(mu) + var_square - 1. - 2 * log_std -1)
        return kl

    def get_batches(self, df, batch_size, user_col, item_col, rating_col, key_col, num_keys):

        remaining_size = len(df)

        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                df_batch = df[batch_index*batch_size:]
                positive_data = get_arrays(df_batch, user_col, item_col, rating_col, key_col, num_keys)
                negative_data = get_negative_sample(df_batch, self.num_items, user_col, item_col, 10, num_keys)
                train_array = concate_data(positive_data, negative_data)
                batches.append(train_array)
            else:
                df_batch = df[batch_index*batch_size:(batch_index+1)*batch_size]
                positive_data = get_arrays(df_batch, user_col, item_col, rating_col, key_col, num_keys)
                negative_data = get_negative_sample(df_batch, self.num_items, user_col, item_col, 10, num_keys)
                train_array = concate_data(positive_data, negative_data)
                batches.append(train_array)
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def train_model(self, df, epoch=100, batches=None,
                    user_col='UserIndex', item_col='ItemIndex', rating_col='Binary', key_col='keyVector', **unused):

        self.create_embeddings(df, user_col, item_col, rating_col)

        if batches is None:
            batches = self.get_batches(df, self.batch_size, user_col, item_col, rating_col, key_col, self.text_dim)

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                data_batch = batches[step]
                user_index = data_batch[0]
                item_index = data_batch[1]
                rating = data_batch[2]
                keyphrase = data_batch[3].todense()
                feed_dict = {self.users_index: user_index, self.items_index: item_index, self.corruption: 0.1,
                             self.rating: rating, self.keyphrase: keyphrase, self.sampling: True}

                training, loss = self.sess.run([self.train, self.loss], feed_dict=feed_dict)
                pbar.set_description("loss:{0}".format(loss))

            #if (i+1) % 5 == 0:
            batches = self.get_batches(df, self.batch_size, user_col, item_col, rating_col, key_col, self.text_dim)

    def predict(self, inputs):
        user_index = inputs[:, 0]
        item_index = inputs[:, 1]
        feed_dict = {self.users_index: user_index, self.items_index: item_index,
                     self.sampling: False, self.corruption: 0}
        return self.sess.run([self.rating_prediction, self.phrase_prediction], feed_dict=feed_dict)

    # def refine_predict(self, inputs, critiqued):
    #     user_index = inputs[:, 0]
    #     item_index = inputs[:, 1]
    #     feed_dict = {self.users_index: user_index,
    #                  self.items_index: item_index,
    #                  self.modified_phrase: critiqued}
    #     modified_rating, modified_phrases = self.sess.run([self.modified_rating_prediction,
    #                                                        self.modified_phrase_prediction],
    #                                                       feed_dict=feed_dict)
    #
    #     return modified_rating, modified_phrases


    def create_embeddings(self, df, user_col, item_col, rating_col):
        R = to_sparse_matrix(df, self.num_users, self.num_items, user_col, item_col, rating_col)
        # user_embedding = to_laplacian(R, self.embed_dim)
        # item_embedding = to_laplacian(R.T, self.embed_dim)
        user_embedding, item_embedding = to_svd(R, self.embed_dim)
        # import ipdb; ipdb.set_trace()
        self.sess.run([self.user_embeddings.assign(user_embedding), self.item_embeddings.assign(item_embedding)])




