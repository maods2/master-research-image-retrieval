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
        "model_name": "phikon2", 
        "model_pretreined": "phikon2",
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
    {"model_name": "resnet_fsl", "model_pretreined": "resnet50"},
    {"model_name": "vit_fsl", "model_pretreined": "vit_base_patch16_224"},
    {"model_name": "dino_fsl", "model_pretreined": "vit_small_patch16_224_dino"},
    {"model_name": "dinov2_fsl", "model_pretreined": "dinov2_vitl14"},
    {"model_name": "uni_fsl", "model_pretreined": "uni"},
    {"model_name": "UNI2-h_fsl", "model_pretreined": "UNI2-h"},
    {"model_name": "phikon_fsl", "model_pretreined": "phikon"},
    {"model_name": "phikon2_fsl", "model_pretreined": "phikon2"},
    {"model_name": "virchow2_fsl", "model_pretreined": "Virchow2"},
]
