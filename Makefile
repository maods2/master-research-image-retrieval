.PHONY: train app lint lint-diff clean retrival-all-models retrival-all-datasets-models

# Default configuration
CONFIG ?= default_train_config.yaml

# ============================
# Training Targets
# ============================
train:
	python3 src/main.py --config configs/$(CONFIG) --pipeline train

train-m:
	python3 src/main.py --config configs/multilabel/train_config.yaml --pipeline train

train-tri-vit-b:
	python3 src/main.py --config configs/triplet/train_vit_config.yaml --pipeline train

train-tri-res:
	python3 src/main.py --config configs/triplet/train_resnet_config.yaml --pipeline train

# ==================================
# Few Shot Learning Training Targets
# ==================================

train-fsl-uni:
	python3 src/main.py --config configs/$(DATASET)/fsl_train/uni_fsl_config.yaml --pipeline train
train-fsl-resnet:
	python3 src/main.py --config configs/$(DATASET)/fsl_train/resnet_fsl_config.yaml --pipeline train

train-all-datasets-models-part1:
	datasets="bracs-resized CRC-VAL-HE-7K-splitted"; \
	models="resnet_fsl dino_fsl dinov2_fsl uni_fsl clip_fsl virchow2_fsl vit_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/fsl_train/$$model\_config.yaml --pipeline train; \
		done; \
	done

train-all-datasets-models-part2:
	datasets="ovarian-cancer-splitted skin-cancer-splitted2"; \
	models="resnet_fsl dino_fsl dinov2_fsl uni_fsl clip_fsl virchow2_fsl vit_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/fsl_train/$$model\_config.yaml --pipeline train; \
		done; \
	done

train-all-datasets-models:
	datasets="bracs-resized CRC-VAL-HE-7K-splitted ovarian-cancer-splitted skin-cancer-splitted"; \
	models="resnet_fsl dino_fsl dinov2_fsl uni_fsl clip_fsl virchow2_fsl vit_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/fsl_train/$$model\_config.yaml --pipeline train; \
		done; \
	done

# =========================================
# Few Shot Learning Training Targets AD-HOC
# =========================================

train-uni-fsl:
	python3 src/main.py --config configs/glomerulo/training/uni_fsl_no_augm_train_config.yaml --pipeline train

train-uni-fsl-aug:
	python3 src/main.py --config configs/glomerulo/training/uni_fsl_train_config.yaml --pipeline train

train-resnet-fsl:
	python3 src/main.py --config configs/glomerulo/training/resnet_fsl_no_augm_train_config.yaml --pipeline train

train-fsl-vit:
	python3 src/main.py --config configs/glomerulo/training/vit_fsl_no_augm_train_config.yaml --pipeline train

# ============================
# Classification Targets
# ============================
test-uni-fsl:
	python3 src/main.py --config configs/glomerulo/fsl_test/uni_fsl_no_augm_train_config.yaml --pipeline test

test-uni-fsl-sc:
	python3 src/main.py --config configs/ovarian-cancer-splitted/fsl_test/uni_fsl_no_augm_train_config.yaml --pipeline test

# ============================
# Retrieval Targets
# ============================
retrieval-vit:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/vit_config.yaml --pipeline test

retrieval-resnet:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/resnet50_config.yaml --pipeline test

retrieval-dino:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/dino_config.yaml --pipeline test

retrieval-dinov2:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/dinov2_config.yaml --pipeline test

retrieval-clip:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/clip_config.yaml --pipeline test

retrieval-uni:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/uni_config.yaml --pipeline test

retrieval-virchow2:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/virchow2_config.yaml --pipeline test

retrieval-uni-fsl:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/uni_fsl_config.yaml --pipeline test

retrieval-resnet-fsl:
	python3 src/main.py --config configs/$(DATASET)/retrieval_test/resnet_fsl_config.yaml --pipeline test

retrival-all-models:
# make retrieval-vit DATASET=$(DATASET)
# make retrieval-resnet DATASET=$(DATASET)
# make retrieval-dino DATASET=$(DATASET)
# make retrieval-dinov2 DATASET=$(DATASET)
# make retrieval-clip DATASET=$(DATASET)
# make retrieval-uni DATASET=$(DATASET)
# make retrieval-virchow2 DATASET=$(DATASET)
	make retrieval-uni-fsl DATASET=$(DATASET)
	make retrieval-resnet-fsl DATASET=$(DATASET)

# Example usage:
# make retrival-all DATASET=ovarian-cancer
# make retrival-all DATASET=bracs-resized

retrival-all-datasets-models:
	for dataset in bracs-resized CRC-VAL-HE-7K-splitted glomerulo ovarian-cancer-splitted skin-cancer-splitted; do \
		make retrival-all-models DATASET=$$dataset; \
	done


# ============================
# Download Datasets
# ============================
download-datasets:
	cd ./datasets && \
	gdown https://drive.google.com/uc?id=1OckKt2r-jyQ_Si4HHRn5V7KmVxlICqRg && \
	unzip final.zip && \
	rm -rf final.zip


# ============================
# Application Target
# ============================
app:
	streamlit run demo/app.py

# ============================
# Linting Targets
# ============================
lint:
	blue ./src ./demo && isort ./src ./demo

lint-diff:
	blue --check --diff ./src ./demo && isort --check --diff ./src ./demo

# ============================
# Cleaning Target
# ============================
clean:
	find . -type d -name "__pycache__" ! -path "./env_tcc_eeg/*" -exec rm -rv {} \;
