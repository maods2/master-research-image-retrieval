.PHONY: train

CONFIG ?= default_train_config.yaml


####
train:
	python3 src/main.py --config configs/$(CONFIG) --pipeline train
train-m:
	python3 src/main.py --config configs/multilabel/train_config.yaml --pipeline train

train-tri-vit-b:
	python3 src/main.py --config configs/triplet/train_vit_config.yaml --pipeline train

train-tri-res:
	python3 src/main.py --config configs/triplet/train_resnet_config.yaml --pipeline train







retrieval-vit:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/default_vit_config.yaml --pipeline test
	
retrieval-resnet:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/default_resnet_config.yaml --pipeline test

retrieval-dino:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/default_dino_config.yaml --pipeline test

retrieval-dinov2:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/default_dinov2_config.yaml --pipeline test

retrieval-clip:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/default_clip_config.yaml --pipeline test

retrieval-uni:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/default_uni_foundation_config.yaml --pipeline test

retrieval-virchow2:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/default_virchow2_foundation_config.yaml --pipeline test

retrival-all:
	make retrieval-vit DATASET=$(DATASET)
	make retrieval-resnet DATASET=$(DATASET)
	make retrieval-dino DATASET=$(DATASET)
	make retrieval-dinov2 DATASET=$(DATASET)
	make retrieval-clip DATASET=$(DATASET)
	make retrieval-uni DATASET=$(DATASET)
	make retrieval-virchow2 DATASET=$(DATASET)

# example: make retrival-all DATASET=ovarian-cancer
# example: make retrival-all DATASET=ovarian-cancer


app:
	streamlit run demo/app.py

lint:
	blue ./src ./demo && isort  ./src ./demo

lint-diff:
	blue --check --diff ./src ./demo && isort --check --diff ./src ./demo

clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;
