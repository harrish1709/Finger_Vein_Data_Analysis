from keras.layers import Layer
import tensorflow as tf

class CapsuleLayer(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]  # 576
        self.input_dim_capsule = input_shape[2]  # 64

        # Transform matrix: (1, input_num_capsule, num_capsule, dim_capsule, input_dim_capsule)
        self.W = self.add_weight(
            shape=[1, self.input_num_capsule, self.num_capsule, self.dim_capsule, self.input_dim_capsule],
            initializer='glorot_uniform',
            trainable=True,
            name='W'
        )

    def call(self, inputs):
        # Expand dims to match W: (batch, input_num_capsule, 1, 1, input_dim_capsule)
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 2)

        # Tile inputs: (batch, input_num_capsule, num_capsule, dim_capsule, input_dim_capsule)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsule, self.dim_capsule, 1])

        # Tile W to match batch: (batch, input_num_capsule, num_capsule, dim_capsule, input_dim_capsule)
        W_tiled = tf.tile(self.W, [tf.shape(inputs)[0], 1, 1, 1, 1])

        # u_hat = matmul(W, inputs)
        u_hat = tf.reduce_sum(W_tiled * inputs_tiled, axis=-1)  # shape: (batch, input_num_capsule, num_capsule, dim_capsule)

        # Routing
        b = tf.zeros_like(u_hat[..., 0])  # shape: (batch, input_num_capsule, num_capsule)

        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)  # along num_capsule
            c_expanded = tf.expand_dims(c, -1)
            s = tf.reduce_sum(c_expanded * u_hat, axis=1)
            v = self.squash(s)
            if i < self.routings - 1:
                v_expanded = tf.expand_dims(v, 1)
                b += tf.reduce_sum(u_hat * v_expanded, axis=-1)

        return v

    def squash(self, s, axis=-1):
        s_squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        scale = s_squared_norm / (1. + s_squared_norm + 1e-9)
        return scale * s / tf.sqrt(s_squared_norm + 1e-9)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
