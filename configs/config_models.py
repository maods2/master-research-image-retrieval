old_models = [
    {"model_name": "resnet50", "model_pretreined": ""},
    {"model_name": "dino", "model_pretreined": "vit_small_patch16_224_dino"},
    {"model_name": "dinov2", "model_pretreined": "dinov2_vitl14"},
    {"model_name": "uni", "model_pretreined": "vit_large_patch16_224"},
    {"model_name": "clip", "model_pretreined": "openai/clip-vit-base-patch32"},
    {"model_name": "virchow2", "model_pretreined": "hf-hub:paige-ai/Virchow2"},
    {"model_name": "vit", "model_pretreined": "vit_base_patch16_224"},
]

retrieval_backbone_models = [
    {
        "model_name": "resnet", 
        "model_pretreined": "resnet50",
        "checkpoint_path": "",
        "load_checkpoint": False,
    },
    {
        "model_name": "vit", 
        "model_pretreined": "vit_base_patch16_224",
        "checkpoint_path": "",
        "load_checkpoint": False,
    },
    {
        "model_name": "dino", 
        "model_pretreined": "vit_small_patch16_224_dino",
        "checkpoint_path": "",
        "load_checkpoint": False,
    },
    {
        "model_name": "dinov2", 
        "model_pretreined": "dinov2_vitl14",
        "checkpoint_path": "",
        "load_checkpoint": False,
    },
    {
        "model_name": "uni", 
        "model_pretreined": "uni",
        "checkpoint_path": "",
        "load_checkpoint": False,
    },
    {
        "model_name": "UNI2-h", 
        "model_pretreined": "UNI2-h",
        "checkpoint_path": "",
        "load_checkpoint": False,
    },
    {
        "model_name": "phikon", 
        "model_pretreined": "phikon",
        "checkpoint_path": "",
        "load_checkpoint": False,
    },
    {
        "model_name": "phikon-v2", 
        "model_pretreined": "phikon-v2",
        "checkpoint_path": "",
        "load_checkpoint": False,
    },
    {
        "model_name": "virchow2", 
        "model_pretreined": "Virchow2",
        "checkpoint_path": "",
        "load_checkpoint": False,
    },
]

fsl_models = [
    # {"model_name": "resnet_fsl", "model_pretreined": "resnet50"},
    # {"model_name": "vit_fsl", "model_pretreined": "vit_base_patch16_224"},
    # {"model_name": "dino_fsl", "model_pretreined": "vit_small_patch16_224_dino"},
    # {"model_name": "dinov2_fsl", "model_pretreined": "dinov2_vitl14"},
    # {"model_name": "uni_fsl", "model_pretreined": "uni"},
    # {"model_name": "UNI2-h_fsl", "model_pretreined": "UNI2-h"},
    {"model_name": "phikon_fsl", "model_pretreined": "phikon"},
    {"model_name": "phikon-v2_fsl", "model_pretreined": "phikon-v2"},
    # {"model_name": "virchow2_fsl", "model_pretreined": "Virchow2"},
]

fsl_test_models = [
    {
        "model_name": "resnet_fsl", 
        "model_pretreined": "resnet50",
        "checkpoint_path": "./artifacts/retr_fsl_train_test_glomerulo/resnet_fsl_glomerulo_resnet_fsl_2025-05-26_02-49-38_checkpoint.pth",
        "load_checkpoint": True,
    },
    {
        "model_name": "vit_fsl", 
        "model_pretreined": "vit_base_patch16_224",
        "checkpoint_path": "./artifacts/retr_fsl_train_test_glomerulo/vit_fsl_glomerulo_vit_fsl_2025-05-26_05-02-00_checkpoint.pth",
        "load_checkpoint": True,
    },
    {
        "model_name": "dino_fsl", 
        "model_pretreined": "vit_small_patch16_224_dino",
        "checkpoint_path": "./artifacts/retr_fsl_train_test_glomerulo/dino_fsl_glomerulo_dino_fsl_2025-05-26_11-52-34_checkpoint.pth",
        "load_checkpoint": True,
    },
    {
        "model_name": "dinov2_fsl", 
        "model_pretreined": "dinov2_vitl14",
        "checkpoint_path": "./artifacts/retr_fsl_train_test_glomerulo/dinov2_fsl_glomerulo_dinov2_fsl_2025-05-26_14-30-20_checkpoint.pth",
        "load_checkpoint": True,
    },
    {
        "model_name": "uni_fsl", 
        "model_pretreined": "uni",
        "checkpoint_path": "./artifacts/retr_fsl_train_test_glomerulo/uni_fsl_glomerulo_uni_fsl_2025-05-27_14-28-01_checkpoint.pth",
        "load_checkpoint": True,
    },
    {
        "model_name": "UNI2-h_fsl", 
        "model_pretreined": "UNI2-h",
        "checkpoint_path": "./artifacts/retr_fsl_train_test_glomerulo/UNI2-h_fsl_glomerulo_UNI2-h_fsl_2025-05-28_09-46-09_checkpoint.pth",
        "load_checkpoint": True,
    },
    # {
    #     "model_name": "phikon_fsl", 
    #     "model_pretreined": "phikon",
    #     "checkpoint_path": "",
    #     "load_checkpoint": False,
    # },
    # {
    #     "model_name": "phikon-v2_fsl", 
    #     "model_pretreined": "phikon-v2",
    #     "checkpoint_path": "",
    #     "load_checkpoint": False,
    # },
    {
        "model_name": "virchow2_fsl", 
        "model_pretreined": "Virchow2",
        "checkpoint_path": "./artifacts/retr_fsl_train_test_glomerulo/virchow2_fsl_glomerulo_virchow2_fsl_2025-05-29_16-05-49_checkpoint.pth",
        "load_checkpoint": True,
    },
]