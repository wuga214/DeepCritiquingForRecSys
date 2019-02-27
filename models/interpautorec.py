import tensorflow as tf
from tqdm import tqdm


class InterpretableAutoRec(object):
    def __init__(self,
                 input_dim,
                 embed_dim,
                 batch_size,
                 lamb=0.01,
                 learning_rate=1e-3,
                 optimizer=tf.train.AdamOptimizer,
                 **unused):
        self.input_dim = self.output_dim = input_dim
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lamb = lamb
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):
        self.inputs = tf.placeholder(tf.float32, (None, self.input_dim[0], self.input_dim[1]))

        x = tf.reshape(tf.transpose(self.inputs, perm=[0, 2, 1]), shape=[-1, 1, self.input_dim[1], self.input_dim[0]])

        with tf.variable_scope('encode'):
            encoded = tf.layers.conv2d(x, self.embed_dim, 1, activation=tf.nn.relu,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb))

        with tf.variable_scope('decode'):
            decoded = tf.layers.conv2d(encoded, self.input_dim[0], 1, activation=None,
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.lamb))

        predict = tf.transpose(tf.reshape(decoded, shape=[-1, self.input_dim[1], self.input_dim[0]]),
                                    perm=[0, 2, 1])

        self.output = tf.nn.sigmoid(predict)

        with tf.variable_scope('loss'):
            l2_loss = tf.losses.get_regularization_loss()
            sigmoid_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.inputs[:, :, 0],
                                                                   logits=predict[:, :, 0])
            self.loss = tf.reduce_mean(sigmoid_loss) + self.lamb * tf.reduce_mean(l2_loss)

        with tf.variable_scope('optimizer'):
            self.train = self.optimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def get_batches(self, rating_tensor, batch_size):
        remaining_size = rating_tensor.shape[0]
        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_tensor[batch_index*batch_size:])
            else:
                batches.append(rating_tensor[batch_index*batch_size:(batch_index+1)*batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def train_model(self, rating_tensor, epoch=100, batches=None, **unused):
        if batches is None:
            batches = self.get_batches(rating_tensor, self.batch_size)

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense()}
                training, loss = self.sess.run([self.train, self.loss], feed_dict=feed_dict)
                pbar.set_description("loss:{0}".format(loss))

    def predict(self, batch_tensor):
        feed_dict = {self.inputs: batch_tensor.todense()}
        return self.sess.run(self.output, feed_dict=feed_dict)

