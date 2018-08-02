import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from preprocessor import maybe_process, vocabularies

print(tf.__version__)
tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

DUMPDIR = "dump"
OUTDIR = os.path.join("trained_models", "Linear")


# Define your feature columns
def create_feature_cols(user_vocabulary):
    return [
        tf.feature_column.categorical_column_with_vocabulary_list(
            "reporter",
            vocabulary_list=user_vocabulary.tolist(),
            default_value=0
        ),

        # tf.feature_column.embedding_column(
        #     tf.feature_column.categorical_column_with_vocabulary_list(
        #         "reporter", vocabulary_list=user_vocabulary.tolist(),
        #         default_value=0),
        #     1500,
        # ),

        hub.text_embedding_column(
            key="description_clean",
            module_spec="https://tfhub.dev/google/Wiki-words-250-with-normalization/1",
            trainable=True,
        ),

        hub.text_embedding_column(
            key="summary_clean",
            module_spec="https://tfhub.dev/google/Wiki-words-250-with-normalization/1",
            trainable=True,
        ),

    ]


def add_more_features(df):
    # TODO: Add more features to the dataframe
    return df


def make_input_fn(df, num_epochs):
    """Create pandas input function"""
    return tf.estimator.inputs.pandas_input_fn(
        x=add_more_features(df),
        y=df["most_active"],
        batch_size=128,
        num_epochs=num_epochs,
        shuffle=True,
        queue_capacity=1000,
        num_threads=1
    )


def train_and_evaluate(output_dir, num_train_steps, assignee_vocabulary, user_vocabulary, traindf, evaldf):
    """Create estimator train and evaluate function"""
    estimator = tf.estimator.LinearClassifier(
        model_dir=output_dir,
        feature_columns=create_feature_cols(user_vocabulary),
        n_classes=len(assignee_vocabulary.tolist()),
        label_vocabulary=assignee_vocabulary.tolist(),
    )

    train_spec = tf.estimator.TrainSpec(input_fn=make_input_fn(traindf, None),
                                        max_steps=num_train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=make_input_fn(evaldf, 1),
                                      steps=None,
                                      start_delay_secs=1,  # start evaluating after N seconds,
                                      throttle_secs=10)  # evaluate every N seconds
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def model(df, force_clean=False):
    user_vocabulary, assignee_vocabulary, most_active_vocabulary = vocabularies(df)

    # Now, split the data into two parts -- training and evaluation.
    np.random.seed(seed=1)  # makes result reproducible
    msk = np.random.rand(len(df)) < 0.8
    traindf = df[msk]
    evaldf = df[~msk]

    # Run the model
    if force_clean:
        shutil.rmtree(OUTDIR, ignore_errors=True)
    train_and_evaluate(OUTDIR, 1000, most_active_vocabulary, user_vocabulary, traindf, evaldf)


if __name__ == "__main__":
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nltk', 'tensorflow-hub'])
    print("loading data")
    data = maybe_process(os.path.join(DUMPDIR, "data.pkl"))
    print("loading data completed")
    print("running model")
    model(data)
    print("running model completed")
