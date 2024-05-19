import jax.numpy as jnp

class NDE:
    def __init__(self, input_layer, recurrent, bias, action_layer, action_bias) -> None:
        self.input_layer = input_layer
        self.recurrent = recurrent
        self.bias = bias
        self.action_layer = action_layer
        self.action_bias = action_bias

    def update(self, input, latent):
        x = jnp.concatenate([input])
        x = jnp.tanh(self.input_layer@x + self.recurrent@latent + self.bias)
        return x
    
    def act(self, latent):

        return self.action_layer@latent + self.action_bias
    
class ParameterReshaper:
    def __init__(self, input_dim, latent_dim, output_dim, n_target):
        self.input_layer_shape = (input_dim + n_target, latent_dim)
        self.recurrent_layer_shape = (latent_dim, latent_dim)
        self.bias_shape = (latent_dim,)

        self.output_layer_shape = (latent_dim + n_target, output_dim)
        self.output_bias_shape = (output_dim,)

        self.total_parameters = jnp.sum(jnp.array([*map(lambda l: l[0]*l[1], [self.input_layer_shape, self.recurrent_layer_shape, self.output_layer_shape])])) \
                            + jnp.sum(jnp.array([*map(lambda l: l[0], [self.bias_shape, self.output_bias_shape])]))

    def __call__(self, params):
        assert params.shape[0] == self.total_parameters

        index = 0
        layers = []

        w = params[index:index+self.input_layer_shape[0]*self.input_layer_shape[1]].reshape(self.input_layer_shape[1], self.input_layer_shape[0])
        index += self.input_layer_shape[0]*self.input_layer_shape[1]
        layers.append(w)

        w = params[index:index+self.recurrent_layer_shape[0]*self.recurrent_layer_shape[1]].reshape(self.recurrent_layer_shape[1], self.recurrent_layer_shape[0])
        index += self.recurrent_layer_shape[0]*self.recurrent_layer_shape[1]
        layers.append(w)

        w = params[index:index+self.bias_shape[0]]
        index += self.bias_shape[0]
        layers.append(w)

        w = params[index:index+self.output_layer_shape[0]*self.output_layer_shape[1]].reshape(self.output_layer_shape[1], self.output_layer_shape[0])
        index += self.output_layer_shape[0]*self.output_layer_shape[1]
        layers.append(w)

        w = params[index:index+self.output_bias_shape[0]]
        index += self.output_bias_shape[0]
        layers.append(w)

        return layers