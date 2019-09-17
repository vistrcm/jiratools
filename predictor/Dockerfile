FROM python:3.7
RUN groupadd -r predictor && useradd --no-log-init -r -g predictor predictor \
    && pip3 --no-cache-dir install torch==1.2.0+cpu torchvision==0.4.0+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip3 --no-cache-dir install fastai flask gunicorn
WORKDIR /app
COPY . /app

USER predictor
EXPOSE 8080
ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8080", "--access-logfile=-", "--log-file=-", "app:app"]
