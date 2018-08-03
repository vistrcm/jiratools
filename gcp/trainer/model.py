from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub

from trainer import labels

CSV_COLUMNS = [
    'id',
    'assignee',
    'key',
    'most_active',
    'reporter',
    'status',
    'summary_clean',
    'description_clean'
]

CSV_COLUMN_DEFAULTS = [
    [0],
    ['unknown_assignee'],
    ['unknown_key'],
    ['very_active_user'],
    ['another_very_active_user'],
    ['completed'],
    ['fix it'],
    ['please fix it']
]

LABEL_COLUMN = 'most_active'
LABELS = labels.LABELS
LABELS_REPORTER = labels.LABELS_REPORTER
# Define the initial ingestion of each feature used by your model.
# Additionally, provide metadata about the feature.
INPUT_COLUMNS = [

    tf.feature_column.categorical_column_with_vocabulary_list(
        "reporter",
        vocabulary_list=LABELS,
        # default_value=0
    ),


    # hub.text_embedding_column(
    #     key="description_clean",
    #     module_spec="https://tfhub.dev/google/Wiki-words-250-with-normalization/1",
    #     trainable=True,
    # ),
    #
    # hub.text_embedding_column(
    #     key="summary_clean",
    #     module_spec="https://tfhub.dev/google/Wiki-words-250-with-normalization/1",
    #     trainable=True,
    # ),
]

UNUSED_COLUMNS = set(CSV_COLUMNS) - {col.name for col in INPUT_COLUMNS} - \
                 {LABEL_COLUMN}


def json_serving_input_fn():
    """Build the servin inputs."""
    inputs = {}
    for feat in INPUT_COLUMNS:
        inputs[feat.name] = tf.placeholder(shape=[None], dtype=feat.dtype)
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


# def example_serving_input_fn():
#     """Build the serving inputs."""
#     example_bytestring = tf.placeholder(
#         shape=[None],
#         dtype=tf.string,
#     )
#     feature_scalars = tf.parse_example(
#         example_bytestring,
#         tf.feature_column.make_parse_example_spec(INPUT_COLUMNS)
#     )
#     return tf.estimator.export.ServingInputReceiver(
#         features,
#         {'example_proto': example_bytestring}
#     )

def csv_serving_input_fn():
    """Build the serving inputs."""
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )
    features = parse_csv(csv_row)
    features.pop(LABEL_COLUMN)
    return tf.estimator.export.ServingInputReceiver(features, {'csv_row': csv_row})


SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    # 'EXAMPLE': example_serving_input_fn,
    'CSV': csv_serving_input_fn
}


def parse_csv(row_string_tensor):
    """Takes the string input tensor and returns a dict of rank-2 tensors.

    Takes a rank-1 tensor and converts it into rank-2 tensor
    example if data is ['csv,line,1', 'csv,line,2', ..] to
    [['csv,line,1'], ['csv,line,2']] which after parsing will result in a
    tuple of tensors: [['csv'], ['csv']], [['line'], ['line']], [[1], [2]]
    """
    row_columns = tf.expand_dims(row_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))

    # remove unused columns
    for col in UNUSED_COLUMNS:
        features.pop(col)
    return features


def parse_label_column(label_string_tensor):
    """Parses a string tensor into the label tensor
    Args:
        label_string_tensor: Tensor of dtype string. Result of parsing the
        CSV column specified by LABEL_COLUMN
    Returns:
        A Tensor of the same shape as label_string_tensor, should return
        an int64 Tensor representing the label index for classification tasks,
        and a float32 Tensor representing the value for a regression task.
    """
    # Build a Hash Table inside the graph
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(LABELS))

    # Use the hash table to convert string labels to ints and one-hot encode
    return table.lookup(label_string_tensor)


def input_fn(filenames,
             num_epochs=None,
             shuffle=True,
             skip_header_lines=0,
             batch_size=200):
    """
    Args:
      filenames: [str] list of CSV files to read data from.
      num_epochs: int how many times through to read the data.
        If None will loop through data indefinitely
      shuffle: bool, whether or not to randomize the order of data.
        Controls randomization of both file order and line order within
        files.
      skip_header_lines: int set to non-zero in order to skip header lines
        in CSV files.
      batch_size: int First dimension size of the Tensors returned by
        input_fn
    Returns:
      A (features, indices) tuple where features is a dictionary of
        Tensors, and indices is a single Tensor of label indices.
    """

    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        # Process the files in a random order.
        filename_dataset = filename_dataset.shuffle(len(filenames))

    # For each filename, parse it into one element per line, and skip the header
    # if necessary
    dataset = filename_dataset.flat_map(
        lambda filename: tf.data.TextLineDataset(filename).skip(skip_header_lines)
    )

    dataset = dataset.map(parse_csv)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features = iterator.get_next()
    return features, parse_label_column(features.pop(LABEL_COLUMN))


def build_estimator(embedding_size, hidden_units, config):
    """Build model"""
    return tf.estimator.LinearClassifier(
        config=config,
        feature_columns=INPUT_COLUMNS,
        n_classes=len(LABELS),
        # label_vocabulary=LABELS,
    )
