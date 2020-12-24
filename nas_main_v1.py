"""
NAS AdaNet

CodeFest, May 2019

Starter code from [AdaNet Tutorial](https://github.com/tensorflow/adanet/blob/master/adanet/examples/tutorials/customizing_adanet_with_tfhub.ipynb)

Firstly download artifacts using the method at https://www.tensorflow.org/hub/common_issues
"""
import re
import os
import shutil
import pickle
import functools
import numpy as np
import pandas as pd

import adanet
import tensorflow as tf
import tensorflow_hub as hub

# from cloud.aws.s3.core import download_dir
#
#

SUB_DIR = "/tmp/subnet/"
if not os.path.exists(SUB_DIR):
    os.makedirs(SUB_DIR)

os.system(f"aws s3 cp --recursive s3://nucleus-chc-preprod-datasciences/users/bmcmahon/nas/tf_hub/subnet/ {SUB_DIR}")

RANDOM_SEED = 21
np.random.seed(RANDOM_SEED)

#@title Parameters
FEATURES_KEY = "sentence"
NUM_CLASSES = 2
LEARNING_RATE = 0.05  #@param {type:"number"}
TRAIN_STEPS = 7500  #@param {type:"integer"}
ADANET_ITERATIONS = 3  #@param {type:"integer"}

LOG_DIR = "/tmp/subnet/"

tf.logging.set_verbosity(tf.logging.INFO)

# MODULES = [hub.load_module_spec(f"/tmp/models/{x}") for x in ["t0_nnlm-en-dim50.tgz","t1_nnlm-en-dim128.tgz","t2_universal-sentence-encoder.tgz"]]
# MODULES = [hub.Module(os.path.join(SUBNET_DIR,"sub",module)) for module in [re.findall(r"google\/(.+)\/\d",module)[0] for module in module_list]]
MODULES = ["/tmp/subnet/nnlm-en-dim128","/tmp/subnet/nnlm-en-dim50","/tmp/subnet/universal-sentence-encoder"]

def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
        return pd.DataFrame.from_dict(data)

def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
    fname="aclImdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True
    )
    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                      "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                      "aclImdb", "test"))
    return train_df, test_df

def make_config(experiment_name):
    # Estimator configuration.
    return tf.estimator.RunConfig(
    save_checkpoints_steps=1000,
    save_summary_steps=1000,
    tf_random_seed=RANDOM_SEED,
    model_dir=os.path.join(LOG_DIR, experiment_name))

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

    def __init__(self, learning_rate, max_iteration_steps, seed=None):
        """Initializes a `Generator` that builds `SimpleNetwork`.

        Args:
          learning_rate: The float learning rate to use.
          max_iteration_steps: The number of steps per iteration.
          seed: The random seed.

        Returns:
          An instance of `Generator`.
        """
        self._seed = seed
        self._dnn_builder_fn = functools.partial(
            SimpleNetworkBuilder,
            learning_rate=learning_rate,
            max_iteration_steps=max_iteration_steps)

    def generate_candidates(self, previous_ensemble, iteration_number,
                          previous_ensemble_reports, all_reports):
        """See `adanet.subnetwork.Generator`."""
        module_index = iteration_number % len(MODULES)
        module_name = MODULES[module_index].split("/")[-2]

        print("generating candidate: %s" % module_name)

        seed = self._seed
        # Change the seed according to the iteration so that each subnetwork
        # learns something different.
        if seed is not None:
          seed += iteration_number
        return [self._dnn_builder_fn(seed=seed,
                                     module_name=module_name,
                                     module=MODULES[module_index])]

train_df, test_df = download_and_load_datasets()
train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], num_epochs=None, shuffle=True)
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], shuffle=False)
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(test_df, test_df["polarity"], shuffle=False)

loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

head = tf.contrib.estimator.binary_classification_head(loss_reduction=loss_reduction)

hub_columns=hub.text_embedding_column(FEATURES_KEY, MODULES[1])

max_iteration_steps = TRAIN_STEPS // ADANET_ITERATIONS

estimator = adanet.Estimator(
    head=head,
    subnetwork_generator=SimpleNetworkGenerator(
        learning_rate=LEARNING_RATE,
        max_iteration_steps=max_iteration_steps,
        seed=RANDOM_SEED),
    max_iteration_steps=max_iteration_steps,
    evaluator=adanet.Evaluator(input_fn=train_input_fn, 
                               steps=10),
    report_materializer=None,
    adanet_loss_decay=.99,
    config=make_config("tfhub"))

results, _ = tf.estimator.train_and_evaluate(
    estimator,
    train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn,
                                      max_steps=TRAIN_STEPS),
    eval_spec=tf.estimator.EvalSpec(input_fn=predict_test_input_fn, 
                                    steps=None))
print("Accuracy:", results["accuracy"])
print("Loss:", results["average_loss"])

