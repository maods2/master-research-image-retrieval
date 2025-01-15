import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_transforms(transform_config):
    """
    Factory function to get transformations based on the provided configuration.
    
    Args:
        transform_config (dict): Configuration for transformations. Possible keys include:
            - "resize" (tuple): Height and width for resizing (default: (224, 224))
            - "horizontal_flip" (bool): Whether to apply random horizontal flip (default: False)
            - "random_brightness_contrast" (bool): Whether to apply random brightness/contrast (default: False)
            - "normalize" (tuple): Mean and std for normalization (default: ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
            - "to_tensor" (bool): Whether to convert the image to a tensor (default: True)
    
    Returns:
        transform (A.Compose): A composed Albumentations transformation pipeline.
    """
    transform_list = []

    # Resize transformation
    if "resize" in transform_config:
        resize_height, resize_width = transform_config["resize"]
        transform_list.append(A.Resize(resize_height, resize_width))

    # Horizontal flip
    if transform_config.get("horizontal_flip", False):
        transform_list.append(A.HorizontalFlip(p=0.5))  # Random horizontal flip
    
    # Random brightness/contrast
    if transform_config.get("random_brightness_contrast", False):
        transform_list.append(A.RandomBrightnessContrast(p=0.2))  # Random brightness/contrast
    
    # Normalization
    if "normalize" in transform_config:
        normalize_mean, normalize_std = tuple(transform_config["normalize"]['mean']), tuple(transform_config["normalize"]['std'])
        transform_list.append(A.Normalize(mean=normalize_mean, std=normalize_std))

    # Convert to tensor
    if transform_config.get("to_tensor", True):
        transform_list.append(ToTensorV2())

    # Return the composed transformation pipeline
    return A.Compose(transform_list)
