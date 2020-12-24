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
import re
import argparse
from ai.etl.base_etl import download_and_load_imdb_datasets
from ai.nas.nas_custom import SimpleNetworkGenerator
from ai.nas.nas_base import make_config, build_AutoEnsembleEstimator
from ai.visualization.architecture_viz import ensemble_architecture

'''
This driver file has working pipeline for taking TF hub embeddings and running them through the pre-made DNN estimator
and then using the AutoEnsembler 
-this mainly focuses on getting the autoensembler to work
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

    if os.path.exists("model_checkpoints"):
        print("Deleting model_checkpoints")
        shutil.rmtree("model_checkpoints")

    #TODO encapsulate this directory structure building part
    SUB_DIR = "/tmp/subnet/"
    if not os.path.exists(SUB_DIR):
        os.makedirs(SUB_DIR)
        os.system(
           f"aws s3 cp --recursive s3://nucleus-chc-preprod-datasciences/users/bmcmahon/nas/tf_hub/subnet/ {SUB_DIR}")

    np.random.seed(args['random_seed'])
    tf.logging.set_verbosity(tf.logging.INFO)

    #ETL for train and test dataset
    train_df, test_df = download_and_load_imdb_datasets()

    train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], num_epochs=None, shuffle=True)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                          max_steps=args['train_steps'])

    #TODO look at usage here. This is not used here right?? This was for example predict part?
    # predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(train_df, train_df["polarity"], shuffle=False)

    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(test_df, test_df["polarity"], shuffle=False)
    eval_spec=tf.estimator.EvalSpec(input_fn=predict_test_input_fn,
                                        steps=None)

    #create head based on type
    loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
    binary_head = tf.contrib.estimator.binary_classification_head(loss_reduction=loss_reduction)

    #TODO add function to determine if using TF hub modules or not and then process data differently

    #build embedding column
    MODULES = ["/tmp/subnet/nnlm-en-dim128", "/tmp/subnet/nnlm-en-dim50", "/tmp/subnet/universal-sentence-encoder"]

    ndim50_embeddings = hub.text_embedding_column(args['text_feature_col'], MODULES[1])

    encoder_embeddings = hub.text_embedding_column(args['text_feature_col'], MODULES[2])

    #build estimator(s)
    max_iteration_steps = args['train_steps'] // args['adanet_iterations']

    # simple_network_estimator = adanet.Estimator(
    #     head=binary_head,
    #     subnetwork_generator=SimpleNetworkGenerator(
    #         feature_columns=[ndim50_embeddings],
    #         learning_rate=args["learning_rate"] ,
    #         max_iteration_steps=max_iteration_steps,
    #         seed=args['random_seed']),
    #     max_iteration_steps=max_iteration_steps,
    #     evaluator=adanet.Evaluator(input_fn=train_input_fn,
    #                                steps=10),
    #     report_materializer=None,
    #     adanet_loss_decay=.99,
    #     config=make_config(args))



    #out of the box estimator
    estimator_ndim = tf.estimator.DNNClassifier(
        #   head=multi_class_head,
        n_classes=args['num_classes'],
        hidden_units=[64, 10],
        feature_columns=[ndim50_embeddings]
    )

    #testing
    estimator_encoder = tf.estimator.DNNClassifier(
        #   head=multi_class_head,
        n_classes=args['num_classes'],
        hidden_units=[64, 10],
        feature_columns=[encoder_embeddings]
    )

    #TODO need to change later to pick up from dynamic list
    candidate_pool_list = [estimator_ndim, estimator_encoder]

    ensemble_estimator = build_AutoEnsembleEstimator(binary_head, candidate_pool_list, args)

    # train and evaluate
    results, _ = tf.estimator.train_and_evaluate(
        ensemble_estimator,
        train_spec=train_spec,
        eval_spec=eval_spec )

    #Model Results and Visualization Module- Gauri, you can take the results callback from train_evaluate and use results
    #TODO separate out this part into visualization package
    print("Accuracy:", results["accuracy"])
    print("Loss:", results["average_loss"])

    #show ensemble results
    ensemble_architecture(results)



if __name__ == "__main__":
    #TODO move this into argparser but putting here for now to be quick in extracting args to be parameterized
    test_args = {
        "dataset_path": "./datasets/imdb_codefest/",
        "text_feature_col": "sentence",
        "num_classes": 2,
        "random_seed": 5,
        "log_dir":  "/tmp/subnet/",
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
