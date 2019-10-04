import argparse
import os
from pathlib import Path

from fastai.text import load_data, TextList, text_classifier_learner, AWD_LSTM
from google.cloud import storage


def download_training(bucket, bucket_path, dst_dir):
    for file in ["models", "classifier.csv", "data_lm.pkl"]:
        path = bucket_path + "/" + file
        dst_path = dst_dir + "/" + file
        download_file(bucket, path, dst_path)

    # show downloaded files
    for root, dirs, files in os.walk("."):
        print(root, dirs, files)


def upload_results(bucket, bucket_path, src):
    for file in ["models", "data_clas.pkl", "losses_001.jpg"]:
        f_src = src + "/" + file
        destination_blob_name = bucket_path + "/" + file
        upload_blob(bucket, f_src, destination_blob_name)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))


def download_file(bucket_name, data_path, dst):
    # Initialise a client
    storage_client = storage.Client()
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the filepath
    blob = bucket.blob(data_path)
    # Download the file to a destination
    blob.download_to_filename(dst)


def train(bs):
    path = Path("./")
    data_lm = load_data(path, 'data_lm.pkl', bs=bs)
    print("data_lm loaded")

    data_clas = (TextList.from_csv(path, 'classifier.csv', cols=["summary", "description"], vocab=data_lm.vocab)
                 .split_from_df(col=3)
                 .label_from_df(cols=0)
                 .databunch(bs=bs))

    print("data_clas loaded")
    data_clas.show_batch()  # not sure how it will work

    data_clas.save('data_clas.pkl')

    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.6)
    learn.load_encoder('fine_tuned_enc')

    lr_estimate = 1.0e-2

    learn.fit_one_cycle(1, lr_estimate, moms=(0.8,0.7))

    learn.save('first')
    losses_fig = learn.recorder.plot_losses(return_fig=True)
    losses_fig.savefig("losses_001.jpg", dpi=600)


def parse_args():
    parser = argparse.ArgumentParser(description='Classificator settings')
    parser.add_argument('--bs', default=60, type=int, help='batch size')
    parser.add_argument('--bucket', default="sv-fastai", type=str, help='data bucket')
    parser.add_argument('--training_path', default="datasets/jiratools/classifier.csv", type=str, help='training data')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    download_training(args.bucket, args.training_path, ".")

    train(args.bs)
    upload_results(args.bucket, args.training_path, ".")
    print("completed")


if __name__ == "__main__":
    main()