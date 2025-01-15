.PHONY: train

CONFIG ?= default_config.yaml

train:
	python main.py --config config/$(CONFIG) --pipeline train