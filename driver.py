from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import re
import shutil
import adanet
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import tensorflow_hub as hub
import urllib
import sys
import argparse
from ai.etl.base_etl import *
from ai.nas.nas_base import *
from ai.visualization.architecture_viz import *

'''
Driver combining custom network generation and auto-ensembling
'''


def input_fn_train():
  dataset = tf.data.Dataset.from_tensor_slices((train_features, train_authors))
  dataset = dataset.repeat().shuffle(100).batch(64)
  iterator = dataset.make_one_shot_iterator()
  data, labels = iterator.get_next()
  return data, labels


def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser(description=
                                """
                                This is a test of the command line argument parser in Python.
                                """,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("required_positional_arg",
                   help="desc")
    p.add_argument("required_int", type=int,
                   help="req number")




def main(args):

    ###TODO placeholder for downloading IMDB dataset from S3

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
        path = args['dataset_path']
        train_df = load_dataset(path + "train/" )
        test_df = load_dataset(path + "test/")
        return train_df, test_df

    tf.logging.set_verbosity(tf.logging.INFO)

    train_df, test_df = download_and_load_datasets()

    train_text = train_df["sentence"]
    test_text = test_df["sentence"]
    train_label = train_df["polarity"]
    test_label = test_df["polarity"]

    # Turn the labels into a one-hot encoding
    encoder = LabelEncoder()
    encoder.fit_transform(np.array(train_label))
    train_encoded = encoder.transform(train_label)
    test_encoded = encoder.transform(test_label)
    print(encoder.classes_)
    num_classes = len(encoder.classes_)

    train_labels = np.array(train_encoded).astype(np.int32)


    loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE

    binary_head = tf.contrib.estimator.binary_classification_head(
        loss_reduction=loss_reduction)
    #
    # hub_columns = hub.text_embedding_column(
    #     key=args["text_feature_col"],
    #     module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    def make_config(args):
        # Estimator configuration.
        return tf.estimator.RunConfig(
            save_checkpoints_steps=1000,
            save_summary_steps=1000,
            tf_random_seed=args['random_seed'],
            model_dir=args['model_dir']
        )


    # Create TF Hub embedding columns using 2 different modules
    ndim_embeddings = hub.text_embedding_column(
        "ndim",
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
        trainable=False
    )
    encoder_embeddings = hub.text_embedding_column(
        "encoder",
        module_spec="https://tfhub.dev/google/universal-sentence-encoder/2", trainable=False)

    # Train input function
    def input_fn_train():
        dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels))
        dataset = dataset.repeat().shuffle(100).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        data, labels = iterator.get_next()
        return data, labels

    def input_fn_eval():
        dataset = tf.data.Dataset.from_tensor_slices((eval_features, test_label))
        dataset = dataset.batch(64)
        iterator = dataset.make_one_shot_iterator()
        data, labels = iterator.get_next()
        return data, labels

    # Define the Estimators we'll be feeding into our AdaNet model
    estimator_ndim = tf.estimator.DNNClassifier(
        #   head=multi_class_head,
        n_classes=num_classes,
        hidden_units=[64, 10],
        feature_columns=[ndim_embeddings]
    )

    estimator_encoder = tf.estimator.DNNClassifier(
        #   head=multi_class_head,
        n_classes=num_classes,
        hidden_units=[64, 10],
        feature_columns=[encoder_embeddings]
    )

    batch_size = 64
    total_steps = 4000

    estimator = adanet.AutoEnsembleEstimator(
        head=binary_head,
        candidate_pool=[
            estimator_ndim,
            estimator_encoder
        ],
        config=tf.estimator.RunConfig(
            save_summary_steps=1000,
            save_checkpoints_steps=1000,
            model_dir=args['model_dir']
        ),
        max_iteration_steps=5000
    )

    train_features = {
      "ndim": train_text,
      "encoder": train_text
    }

    eval_features = {
        "ndim": test_text,
        "encoder": test_text
    }


    train_spec = tf.estimator.TrainSpec(
      input_fn=input_fn_train,
      max_steps=40000
    )

    eval_spec=tf.estimator.EvalSpec(
      input_fn=input_fn_eval,
      steps=None,
      start_delay_secs=10,
      throttle_secs=10
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    #TODO move this into argparser but putting here for now to be quick in extracting args to be parameterized
    test_args = {
        "dataset_path": "./datasets/imdb_codefest/",
        "text_feature_col": "sentence",
        "num_classes": 2,
        "random_seed": 5,
        "log_dir":  "./",
        "model_dir": "./model_checkpoints/",
        "experiment_name":  "MVP_run",
        "learning_rate": 0.05,
        "train_steps": 7500,
        "adanet_iterations": 3


    }

    if sys.version_info < (3, 6, 0):
        sys.stderr.write("You need python 3.0 or later to run this script\n")
        sys.exit(1)

    try:
        args = cmdline_args()
        print(args)

    except:
        print('Try $python <script_name> "Hello" 123 --enable')

    #args = args.update(test_args)
    args = test_args
    main(args)
