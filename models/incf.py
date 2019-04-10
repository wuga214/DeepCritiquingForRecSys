from tqdm import tqdm
from utils.reformat import to_sparse_matrix, to_laplacian, to_svd
import numpy as np
import tensorflow as tf
import random


class INCF(object):
    def __init__(self,
                 num_users,
                 num_items,
                 text_dim,
                 embed_dim,
                 num_layers,
                 batch_size,
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
        self.batch_size = batch_size
        self.negative_sampler = negative_sampler
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # tf.summary.FileWriter('./graphs', self.sess.graph)

    def get_graph(self):

        self.users_index = tf.placeholder(tf.int32, [None], name='user_id')
        self.items_index = tf.placeholder(tf.int32, [None], name='item_id')
        self.rating = tf.placeholder(tf.int32, [None], name='rating')
        self.keyphrase = tf.placeholder(tf.int32, [None, self.text_dim], name='key_phrases')
        self.modified_phrase = tf.placeholder(tf.float32, [None, self.text_dim], name='modified_phrases')

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
                #hi = tf.concat([hi, ho], axis=1)
                hi = ho

        with tf.variable_scope("prediction", reuse=False):
            rating_prediction = tf.layers.dense(inputs=hi, units=1,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                                activation=None, name='rating_prediction')
            phrase_prediction = tf.layers.dense(inputs=hi, units=self.text_dim,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb),
                                                activation=None, name='phrase_prediction')

            self.rating_prediction = rating_prediction
            self.phrase_prediction = phrase_prediction

        with tf.variable_scope("losses"):
            phrase_condition = tf.stop_gradient(tf.cast(tf.reduce_max(self.keyphrase, axis=1), tf.float32))

            with tf.variable_scope("rating_loss"):
                # rating_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.reshape(self.rating, [-1, 1]),
                #                                               logits=self.rating_prediction)
                rating_loss = tf.losses.mean_squared_error(labels=tf.reshape(self.rating, [-1, 1]),
                                                           predictions=self.rating_prediction)

            with tf.variable_scope("phrase_loss"):
                phrase_loss = tf.losses.mean_squared_error(labels=self.keyphrase,
                                                           predictions=self.phrase_prediction) * phrase_condition

            with tf.variable_scope("l2"):
                l2_loss = tf.losses.get_regularization_loss()

            self.loss = (tf.reduce_mean(rating_loss)
                         + tf.reduce_mean(phrase_loss)
                         + l2_loss
                         )

        with tf.variable_scope('optimizer'):
            self.train = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train_model(self, df, epoch=100, batches=None,
                    user_col='UserIndex', item_col='ItemIndex', rating_col='Binary', key_col='keyVector', **unused):

        self.create_embeddings(df, user_col, item_col, rating_col)

        if batches is None:
            batches = self.negative_sampler.get_batches()

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                data_batch = batches[step]
                user_index = data_batch[0]
                item_index = data_batch[1]
                rating = data_batch[2]
                keyphrase = data_batch[3].todense()
                feed_dict = {self.users_index: user_index, self.items_index: item_index,
                             self.rating: rating, self.keyphrase: keyphrase}

                training, loss = self.sess.run([self.train, self.loss], feed_dict=feed_dict)
                pbar.set_description("loss:{0}".format(loss))

            #if (i+1) % 5 == 0:
            batches = self.negative_sampler.get_batches()

    def predict(self, inputs):
        user_index = inputs[:, 0]
        item_index = inputs[:, 1]
        feed_dict = {self.users_index: user_index, self.items_index: item_index}
        return self.sess.run([self.rating_prediction, self.phrase_prediction], feed_dict=feed_dict)

    def create_embeddings(self, df, user_col, item_col, rating_col):
        R = to_sparse_matrix(df, self.num_users, self.num_items, user_col, item_col, rating_col)
        user_embedding, item_embedding = to_svd(R, self.embed_dim)
        self.sess.run([self.user_embeddings.assign(user_embedding), self.item_embeddings.assign(item_embedding)])

    def save_model(self, path, name):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "{0}/{1}/model.ckpt".format(path, name))
        print("Model saved in path: %s" % save_path)

    def load_model(self, path, name):
        saver = tf.train.Saver()
        saver.restore(self.sess, "{0}/{1}/model.ckpt".format(path, name))

