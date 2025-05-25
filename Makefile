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

# CRC-VAL-HE-7K-splitted - clip_fsl virchow2_fsl vit_fsl
# not used - uni_fsl CRC-VAL-HE-7K-splitted 
train-all-datasets-models-part1:
	datasets="bracs-resized"; \
	models="resnet_fsl dino_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/fsl_train/$$model\_config.yaml --pipeline train; \
		done; \
	done

# ovarian-cancer-splitted - ALREADY TRAINED
# skin-cancer-splitted - resnet_fsl dino_fsl dinov2_fsl
# not used - uni_fsl
train-all-datasets-models-part2:
	datasets="skin-cancer-splitted CRC-VAL-HE-7K-splitted bracs-resized"; \
	models="virchow2_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/fsl_train/$$model\_config.yaml --pipeline train; \
		done; \
	done

train-all-datasets-models-part3:
	datasets="glomerulo"; \
	models="virchow2_fsl resnet_fsl clip_fsl vit_fsl dino_fsl dinov2_fsl uni_fsl"; \
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

make-test:
	datasets="ovarian-cancer-splitted"; \
	models="resnet_fsl uni_fsl uni_fsl2 virchow2_fsl philkon_fsl philkon_fsl2"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Testing on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/fsl_test/$$model\_config.yaml --pipeline test; \
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
retrieval-test-pretrained1:
	datasets="glomerulo ovarian-cancer-splitted skin-cancer-splitted"; \
	models="resnet vit dino dinov2 uni UNI2-h philkon philkon2 virchow2"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_test_backone/$$model\_config.yaml --pipeline test; \
		done; \
	done

retrieval-test-pretrained2:
	datasets="glomerulo ovarian-cancer-splitted skin-cancer-splitted"; \
	models="resnet vit dino dinov2 uni UNI2-h philkon philkon2 virchow2"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_test_backone_norm/$$model\_config.yaml --pipeline test; \
		done; \
	done


# ============================
# Download Datasets
# ============================
download-datasets:
	cd ./datasets && \
	gdown https://drive.google.com/uc?id=14GaaCw7og5jqwsBggb52EgVQKBwOJQpV && \
	unzip final_v2.zip && \
	rm -rf final_v2.zip


# ============================
# Download Models
# ============================
download-models:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=1kBwDBUA85wo7IQS54WFcBxHnKd2cHbOg && \
	gdown https://drive.google.com/uc?id=1FisQEXGLm5e0gWE2o0-jxWuef757GtDJ && \
	gdown https://drive.google.com/uc?id=1NV4dKyaOVmMtr-P_YP_p58KTLZqX_GZJ && \
	gdown https://drive.google.com/uc?id=1ip3sTjGoMWpGfcheNpaizhbLHQqUpDtM && \
	gdown https://drive.google.com/uc?id=12jdPlh2gDTVZflMc8SEzPos1XRDTains && \
	unzip Virchow2.zip && \
	rm -rf Virchow2.zip \
	unzip UNI2-h.zip && \
	rm -rf UNI2-h.zip \
	unzip uni.zip && \
	rm -rf uni.zip \
	unzip phikon-v2.zip && \
	rm -rf phikon-v2.zip \
	unzip phikon.zip && \
	rm -rf phikon.zip 

# ============================
# Huggingface Model Login
# ============================

hf-login:
	huggingface-cli login

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
