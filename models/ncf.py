from tqdm import tqdm
from utils.reformat import to_sparse_matrix, to_svd

import tensorflow as tf


class NCF(object):
    def __init__(self,
                 num_users,
                 num_items,
                 text_dim,
                 embed_dim,
                 num_layers,
                 negative_sampler,
                 lamb=0.01,
                 learning_rate=1e-4,
                 optimizer=tf.train.AdamOptimizer,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.negative_sampler = negative_sampler
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # tf.summary.FileWriter('./graphs', self.sess.graph)

    def get_graph(self):

        self.users_index = tf.placeholder(tf.int32, [None], name='user_id')
        self.items_index = tf.placeholder(tf.int32, [None], name='item_id')
        self.rating = tf.placeholder(tf.int32, [None], name='rating')
        self.keyphrase_vector = tf.placeholder(tf.int32, [None, self.text_dim], name='keyphrases_vector')
        self.modified_keyphrase = tf.placeholder(tf.float32, [None, self.text_dim], name='modified_keyphrases')

        with tf.variable_scope("embeddings"):
            self.user_embeddings = tf.Variable(tf.random_normal([self.num_users, self.embed_dim],
                                                                stddev=1 / (self.embed_dim ** 0.5),
                                                                dtype=tf.float32), trainable=True)

            self.item_embeddings = tf.Variable(tf.random_normal([self.num_items, self.embed_dim],
                                                                stddev=1 / (self.embed_dim ** 0.5),
                                                                dtype=tf.float32), trainable=True)

            users = tf.nn.embedding_lookup(self.user_embeddings, self.users_index, name="user_lookup")
            items = tf.nn.embedding_lookup(self.item_embeddings, self.items_index, name="item_lookup")

        with tf.variable_scope("residual"):
            hi = tf.concat([users, items], axis=1)
            for i in range(self.num_layers):
                ho = tf.layers.dense(inputs=hi, units=self.embed_dim*2,
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                     activation=tf.nn.relu)
                hi = ho

        with tf.variable_scope("prediction", reuse=False):
            rating_prediction = tf.layers.dense(inputs=hi, units=1,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                                activation=None, name='rating_prediction')
            keyphrase_prediction = tf.layers.dense(inputs=hi, units=self.text_dim,
                                                   activation=None, name='keyphrase_prediction')

            self.rating_prediction = rating_prediction
            self.keyphrase_prediction = keyphrase_prediction

        with tf.variable_scope("losses"):

            with tf.variable_scope("rating_loss"):
                rating_loss = tf.losses.mean_squared_error(labels=tf.reshape(self.rating, [-1, 1]),
                                                           predictions=self.rating_prediction)

            with tf.variable_scope("l2"):
                l2_loss = tf.losses.get_regularization_loss()

            self.loss = (tf.reduce_mean(rating_loss)
                         + l2_loss
                         )

        with tf.variable_scope('optimizer'):
            self.train = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train_model(self, df, user_col, item_col, rating_col, epoch=100,
                    batches=None, init_embedding=True, **unused):

        if init_embedding:
            self.get_user_item_embeddings(df, user_col, item_col, rating_col)

        if batches is None:
            batches = self.negative_sampler.get_batches()

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for batch in batches:
                feed_dict = {self.users_index: batch[0],
                             self.items_index: batch[1],
                             self.rating: batch[2],
                             self.keyphrase_vector: batch[3].todense()}

                training, loss = self.sess.run([self.train, self.loss], feed_dict=feed_dict)
                pbar.set_description("loss:{}".format(loss))

            #if (i+1) % 5 == 0:
            batches = self.negative_sampler.get_batches()

    def predict(self, inputs):
        user_index = inputs[:, 0]
        item_index = inputs[:, 1]
        feed_dict = {self.users_index: user_index,
                     self.items_index: item_index}
        return self.sess.run([self.rating_prediction,
                              self.keyphrase_prediction],
                             feed_dict=feed_dict)

    def get_user_item_embeddings(self, df, user_col, item_col, rating_col):
        R = to_sparse_matrix(df, self.num_users, self.num_items, user_col, item_col, rating_col)
        user_embedding, item_embedding = to_svd(R, self.embed_dim)
        self.sess.run([self.user_embeddings.assign(user_embedding),
                       self.item_embeddings.assign(item_embedding)])

    def save_model(self, path, name):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "{}/{}/model.ckpt".format(path, name))
        print("Model saved in path: %s" % save_path)

    def load_model(self, path, name):
        saver = tf.train.Saver()
        saver.restore(self.sess, "{}/{}/model.ckpt".format(path, name))
        print("Model restored.")

