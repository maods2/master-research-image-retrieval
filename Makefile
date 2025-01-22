.PHONY: train

CONFIG ?= default_config.yaml

train:
	python3 src/main.py --config configs/$(CONFIG) --pipeline train