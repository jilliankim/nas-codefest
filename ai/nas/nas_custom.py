import tensorflow as tf
import adanet
import tensorflow_hub as hub
import functools

#TODO pass in args for modules so you can alter list
MODULES = ["/tmp/subnet/nnlm-en-dim128","/tmp/subnet/nnlm-en-dim50","/tmp/subnet/universal-sentence-encoder"]

class SimpleNetworkBuilder(adanet.subnetwork.Builder):
    """Builds a simple subnetwork with text embedding module."""

    def __init__(self, learning_rate, max_iteration_steps, seed,
               module_name, module):
        """Initializes a `SimpleNetworkBuilder`.

        Args:
          learning_rate: The float learning rate to use.
          max_iteration_steps: The number of steps per iteration.
          seed: The random seed.

        Returns:
          An instance of `SimpleNetworkBuilder`.
        """
        self._learning_rate = learning_rate
        self._max_iteration_steps = max_iteration_steps
        self._seed = seed
        self._module_name = module_name
        self._module = module

    def build_subnetwork(self,
                       features,
                       logits_dimension,
                       training,
                       iteration_step,
                       summary,
                       previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""
        sentence = features["sentence"]
        # Load module and apply text embedding, setting trainable=True.
        m = hub.Module(self._module, trainable=True)
        x = m(sentence)
        kernel_initializer = tf.keras.initializers.he_normal(seed=self._seed)

        # The `Head` passed to adanet.Estimator will apply the softmax activation.
        logits = tf.layers.dense(
            x, units=1, activation=None, kernel_initializer=kernel_initializer)

        # Use a constant complexity measure, since all subnetworks have the same
        # architecture and hyperparameters.
        complexity = tf.constant(1)

        return adanet.Subnetwork(
            last_layer=x,
            logits=logits,
            complexity=complexity,
            persisted_tensors={})

    def build_subnetwork_train_op(self,
                                subnetwork,
                                loss,
                                var_list,
                                labels,
                                iteration_step,
                                summary,
                                previous_ensemble=None):
        """See `adanet.subnetwork.Builder`."""

        learning_rate = tf.train.cosine_decay(
            learning_rate=self._learning_rate,
            global_step=iteration_step,
            decay_steps=self._max_iteration_steps)
        optimizer = tf.train.MomentumOptimizer(learning_rate, .9)
        # NOTE: The `adanet.Estimator` increments the global step.
        return optimizer.minimize(loss=loss, var_list=var_list)

    def build_mixture_weights_train_op(self, loss, var_list, logits, labels,
                                     iteration_step, summary):
        """See `adanet.subnetwork.Builder`."""
        return tf.no_op("mixture_weights_train_op")

    @property
    def name(self):
        """See `adanet.subnetwork.Builder`."""
        return self._module_name


class SimpleNetworkGenerator(adanet.subnetwork.Generator):
    """Generates a `SimpleNetwork` at each iteration.
    """

    def __init__(self, feature_columns, learning_rate, max_iteration_steps, seed=None):
        """Initializes a `Generator` that builds `SimpleNetwork`.

        Args:
          learning_rate: The float learning rate to use.
          max_iteration_steps: The number of steps per iteration.
          seed: The random seed.

        Returns:
          An instance of `Generator`.
        """
        self._seed = seed
        self._feature_columns = feature_columns
        self._dnn_builder_fn = functools.partial(
            SimpleNetworkBuilder,
            learning_rate=learning_rate,
            max_iteration_steps=max_iteration_steps)

    ## Removed the module iteration b/c swapping out modules makes it more complicated, probably better focus on architectures changing first