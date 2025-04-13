.PHONY: train

CONFIG ?= default_train_config.yaml



train:
	python3 src/main.py --config configs/$(CONFIG) --pipeline train
train-m:
	python3 src/main.py --config configs/multilabel/train_config.yaml --pipeline train

train-tri-vit-b:
	python3 src/main.py --config configs/triplet/train_vit_config.yaml --pipeline train

train-tri-res:
	python3 src/main.py --config configs/triplet/train_resnet_config.yaml --pipeline train

retrieval-vit:
	python3 src/main.py --config configs/retrieval_test/default_vit_config.yaml --pipeline test
	
retrieval-resnet:
	python3 src/main.py --config configs/retrieval_test/default_resnet_config.yaml --pipeline test

retrieval-dino:
	python3 src/main.py --config configs/retrieval_test/default_dino_config.yaml --pipeline test

retrieval-dinov2:
	python3 src/main.py --config configs/retrieval_test/default_dinov2_config.yaml --pipeline test

retrieval-clip:
	python3 src/main.py --config configs/retrieval_test/default_clip_config.yaml --pipeline test

retrieval-uni:
	python3 src/main.py --config configs/retrieval_test/default_uni_foundation_config.yaml --pipeline test

retrieval-virchow2:
	python3 src/main.py --config configs/retrieval_test/default_virchow2_foundation_config.yaml --pipeline test

retrival-all:
	make retrieval-vit
	make retrieval-resnet
	make retrieval-dino
	make retrieval-dinov2
	make retrieval-clip
	make retrieval-uni
	make retrieval-virchow2

app:
	streamlit run demo/app.py

lint:
	blue ./src ./demo && isort  ./src ./demo

lint-diff:
	blue --check --diff ./src ./demo && isort --check --diff ./src ./demo

clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;
