import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        self.queries = self.add_weight(shape=(input_shape[-1], self.key_dim * self.num_heads),
                                       initializer='glorot_uniform',
                                       trainable=True,
                                       name='queries')
        self.keys = self.add_weight(shape=(input_shape[-1], self.key_dim * self.num_heads),
                                    initializer='glorot_uniform',
                                    trainable=True,
                                    name='keys')
        self.values = self.add_weight(shape=(input_shape[-1], self.key_dim * self.num_heads),
                                      initializer='glorot_uniform',
                                      trainable=True,
                                      name='values')

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        queries = tf.matmul(inputs, self.queries)
        keys = tf.matmul(inputs, self.keys)
        values = tf.matmul(inputs, self.values)

        queries = tf.reshape(queries, (batch_size, -1, self.num_heads, self.key_dim))
        keys = tf.reshape(keys, (batch_size, -1, self.num_heads, self.key_dim))
        values = tf.reshape(values, (batch_size, -1, self.num_heads, self.key_dim))

        attention_scores = tf.matmul(queries, keys, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        output = tf.matmul(attention_weights, values)
        output = tf.reshape(output, (batch_size, -1, self.num_heads * self.key_dim))

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.num_heads * self.key_dim)