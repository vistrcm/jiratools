import argparse
import os

from google.cloud import storage


def download_training(bucket_name, data_path, dst):
    # Initialise a client
    storage_client = storage.Client()
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object from the filepath
    blob = bucket.blob(data_path)
    # Download the file to a destination
    blob.download_to_filename(dst)

def train():
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Classificator settings')
    parser.add_argument('--bs', default=60, type=int, help='batch size')
    parser.add_argument('--bucket', default="sv-fastai", type=str, help='data bucket')
    parser.add_argument('--training_path', default="datasets/jiratools/classifier.csv", type=str, help='training data')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    download_training(args.bucket, args.training_path, "training.csv")
    for root, dirs, files in os.walk("."):
        print(root, dirs, files)


if __name__ == "__main__":
    main()