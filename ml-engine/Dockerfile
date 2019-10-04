FROM gcr.io/sv-fastai/fastai:v1

WORKDIR /root

# Copies the trainer code
RUN mkdir /root/trainer
COPY trainer/classification.py /root/trainer/classification.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "trainer/classification.py"]






