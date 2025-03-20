.PHONY: train

CONFIG ?= default_train_config.yaml

train:
	python3 src/main.py --config configs/$(CONFIG) --pipeline train
train-m:
	python3 src/main.py --config configs/multilabel/train_config.yaml --pipeline train

train-t:
	python3 src/main.py --config configs/triplet_res_train_config.yaml --pipeline train

app:
	streamlit run demo/app.py 
lint:
	blue ./src ./demo && isort  ./src ./demo

lint-diff:
	blue --check --diff ./src ./demo && isort --check --diff ./src ./demo

clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;
