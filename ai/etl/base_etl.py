import os
import re
import tensorflow as tf
import pandas as pd



##TODO these are not generalized and are just for IMDB

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

def download_and_load_imdb_datasets(force_download=False):

    if os.path.exists("./datasets/imdb_codefest/"):
        dataset = "./datasets/imdb_codefest/"
    else:
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


def setup_folder_structure():
    pass