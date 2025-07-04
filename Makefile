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


# models="resnet_fsl vit_fsl dino_fsl dinov2_fsl uni_fsl UNI2-h_fsl phikon_fsl phikon-v2_fsl virchow2_fsl"; 
train-fsl-glomerulo:
	datasets="glomerulo"; \
	models="phikon_fsl phikon-v2_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_fsl_train_test/$$model\_config.yaml --pipeline train; \
		done; \
	done

train-fsl-ovarian-cancer:
	datasets="ovarian-cancer-splitted"; \
	models="phikon_fsl phikon-v2_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_fsl_train_test/$$model\_config.yaml --pipeline train; \
		done; \
	done

train-fsl-skin-cancer1:
	datasets="skin-cancer-splitted"; \
	models="phikon_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_fsl_train_test/$$model\_config.yaml --pipeline train; \
		done; \
	done

train-fsl-skin-cancer2:
	datasets="skin-cancer-splitted"; \
	models="phikon-v2_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Training on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_fsl_train_test/$$model\_config.yaml --pipeline train; \
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
# glomerulo   
# phikon phikon2 virchow2
retrieval-test-pretrained1:
	datasets="ovarian-cancer-splitted"; \
	models="phikon-v2"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Retrieve on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_test_backone/$$model\_config.yaml --pipeline test; \
		done; \
	done

# glomerulo 
# UNI2-h phikon phikon2 virchow2
retrieval-test-pretrained2:
	datasets="ovarian-cancer-splitted skin-cancer-splitted"; \
	models="resnet vit dino dinov2 uni UNI2-h phikon phikon-v2 virchow2"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Retrieve on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_test_backone_norm/$$model\_config.yaml --pipeline test; \
		done; \
	done

retrieval-test-fsl-ovarian-cancer:
	datasets="ovarian-cancer-splitted"; \
	models="resnet_fsl vit_fsl dino_fsl dinov2_fsl uni_fsl UNI2-h_fsl phikon-v2_fsl phikon_fsl virchow2_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Retrieve on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_fsl_train_test/$$model\_config.yaml --pipeline test; \
		done; \
	done

# models="vit_fsl dino_fsl dinov2_fsl virchow2_fsl uni_fsl UNI2-h_fsl phikon-v2_fsl phikon_fsl"; 
retrieval-test-fsl-glomerulo:
	datasets="glomerulo"; \
	models="virchow2_fsl"; \
	for dataset in $$datasets; do \
		for model in $$models; do \
			echo "Retrieve on $$dataset with $$model"; \
			python3 src/main.py --config configs/$$dataset/retr_fsl_train_test/$$model\_config.yaml --pipeline test; \
		done; \
	done

# =============================
# Supervised Constrastive Learning
# =============================
train-supcon-glomerulo:
	python3 src/main.py --config configs/glomerulo/sup_con/default_train_config.yaml --pipeline train

train-supcon-ovarian-cancer:
	python3 src/main.py --config configs/ovarian-cancer-splitted/sup_con/default_train_config.yaml--pipeline train

train-supcon-skin-cancer:	
	python3 src/main.py --config configs/skin-cancer-splitted/sup_con/default_train_config.yaml --pipeline train

train-supcon:
	python3 src/main.py --config configs/glomerulo/sup_con/default_train_config.yaml --pipeline train
	python3 src/main.py --config configs/ovarian-cancer-splitted/sup_con/default_train_config.yaml--pipeline train
	python3 src/main.py --config configs/skin-cancer-splitted/sup_con/default_train_config.yaml --pipeline train

# ============================
# Triplet Learning 
# ============================
train-triplet-glomerulo:
	python3 src/main.py --config configs/glomerulo/triplet/default_train_config.yaml --pipeline train

train-triplet-ovarian-cancer:
	python3 src/main.py --config configs/ovarian-cancer-splitted/triplet/default_train_config.yaml --pipeline train

train-triplet-skin-cancer:
	python3 src/main.py --config configs/skin-cancer-splitted/triplet/default_train_config.yaml --pipeline train

train-triplet:
	python3 src/main.py --config configs/glomerulo/triplet/default_train_config.yaml --pipeline train
	python3 src/main.py --config configs/ovarian-cancer-splitted/triplet/default_train_config.yaml --pipeline train
	python3 src/main.py --config configs/skin-cancer-splitted/triplet/default_train_config.yaml --pipeline train

# ============================
# Autoencoder Learning
# ============================
train-autoencoder-glomerulo:
	python3 src/main.py --config configs/glomerulo/autoencoder/default_train_config.yaml --pipeline train
	
train-autoencoder-ovarian-cancer:
	python3 src/main.py --config configs/ovarian-cancer-splitted/autoencoder/default_train_config.yaml --pipeline train

train-autoencoder-skin-cancer:
	python3 src/main.py --config configs/skin-cancer-splitted/autoencoder/default_train_config.yaml --pipeline train

train-autoencoder:
	make train-autoencoder-glomerulo
	make train-autoencoder-ovarian-cancer
	make train-autoencoder-skin-cancer



# ============================
# Retrival embeddings calculation best models
# ============================
retrival-best-models-fsl:
	python3 src/main.py --config configs/glomerulo/retr_fsl_train_test/best_model_test_uni_fsl_config.yaml --pipeline test
	python3 src/main.py --config configs/skin-cancer-splitted/retr_fsl_train_test/best_model_test_phikon-v2_fsl_config.yaml --pipeline test
	python3 src/main.py --config configs/ovarian-cancer-splitted/retr_fsl_train_test/best_model_test_UNI2-h_fsl_config.yaml --pipeline test

retrival-best-models-pt:
	python3 src/main.py --config configs/glomerulo/retr_test_backone/uni_config.yaml --pipeline test
	python3 src/main.py --config configs/skin-cancer-splitted/retr_test_backone/phikon-v2_config.yaml --pipeline test
	python3 src/main.py --config configs/ovarian-cancer-splitted/retr_test_backone/UNI2-h_config.yaml --pipeline test



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
download-virchow2:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=1kBwDBUA85wo7IQS54WFcBxHnKd2cHbOg && \
	unzip Virchow2.zip && \
	rm -rf Virchow2.zip
download-uni2-h:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=1FisQEXGLm5e0gWE2o0-jxWuef757GtDJ && \
	unzip UNI2-h.zip && \
	rm -rf UNI2-h.zip
download-uni:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=1NV4dKyaOVmMtr-P_YP_p58KTLZqX_GZJ && \
	unzip uni.zip && \
	rm -rf uni.zip
download-phikon-v2:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=1ip3sTjGoMWpGfcheNpaizhbLHQqUpDtM && \
	unzip phikon-v2.zip && \
	rm -rf phikon-v2.zip
download-phikon:
	mkdir -p ./assets && \
	cd ./assets && \
	gdown https://drive.google.com/uc?id=12jdPlh2gDTVZflMc8SEzPos1XRDTains && \
	unzip phikon.zip && \
	rm -rf phikon.zip

download-models:
	# make download-virchow2
	make download-uni2-h
	make download-uni
	make download-phikon-v2
	make download-phikon

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
