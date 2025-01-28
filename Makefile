.PHONY: train

CONFIG ?= default_train_config.yaml

train:
	python3 src/main.py --config configs/$(CONFIG) --pipeline train
train-m:
	python3 src/main.py --config configs/multilable_train_config.yaml --pipeline train

app:
	streamlit run demo/app.py 