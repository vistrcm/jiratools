FROM python:3.7-alpine
RUN pip install requests
COPY crawler.py /app/
ENTRYPOINT ["python", "/app/crawler.py"]
