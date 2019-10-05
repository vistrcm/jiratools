BASE_IMAGE_ID = gcr.io/sv-fastai/fastai:v1
IMAGE_ID = gcr.io/sv-fastai/jiratools:v1
EPOCH = $(shell date +%s)
JOB_NAME = classify_$(EPOCH)

.PHONY:build-base
build-base:
	docker build -f Dockerfile-base -t $(BASE_IMAGE_ID) .

.PHONY:push-base
push-base:
	docker push $(BASE_IMAGE_ID)

.PHONY:build
build:
	docker build -t $(IMAGE_ID) .

.PHONY:push
push:
	docker push $(IMAGE_ID)

.PHONY: submit
submit:
	gcloud ai-platform jobs submit training $(JOB_NAME) \
		--region us-central1 \
		--master-image-uri $(IMAGE_ID) \
		-- \
		--bs 60

.PHONY: submitgpu
submitgpu:
	gcloud ai-platform jobs submit training $(JOB_NAME) \
    		--region us-central1 \
    		--master-machine-type n1-highmem-8 \
    		--master-accelerator count=1,type=nvidia-tesla-p4 \
    		--master-image-uri $(IMAGE_ID) \
    		-- \
    		--bs 60
