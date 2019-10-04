IMAGE_ID = gcr.io/sv-fastai/jiratools:v0
EPOCH = $(shell date +%s)
JOB_NAME = classify_$(EPOCH)

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
		--unusedargument

.PHONY: submitgpu
submitgpu:
	gcloud ai-platform jobs submit training $(JOB_NAME) \
    		--region us-central1 \
    		--master-machine-type n1-highmem-8 \
    		--master-accelerator count=1,type=nvidia-tesla-p4 \
    		--master-image-uri $(IMAGE_ID) \
    		-- \
    		--unusedargument