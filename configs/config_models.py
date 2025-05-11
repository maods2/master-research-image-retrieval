retrieval_models = [
    {"model_name": "resnet50", "model_pretreined": ""},
    {"model_name": "dino", "model_pretreined": "vit_small_patch16_224_dino"},
    {"model_name": "dinov2", "model_pretreined": "dinov2_vitl14"},
    {"model_name": "uni", "model_pretreined": "vit_large_patch16_224"},
    {"model_name": "clip", "model_pretreined": "openai/clip-vit-base-patch32"},
    {"model_name": "virchow2", "model_pretreined": "hf-hub:paige-ai/Virchow2"},
    {"model_name": "vit", "model_pretreined": "vit_base_patch16_224"},
]

fsl_models = [
    {
        "model_name": "uni_fsl",
        "model_pretreined": "vit_large_patch16_224",

    },
    {
        "model_name": "resnet_fsl",
        "model_pretreined": "resnet50",

    },
]
