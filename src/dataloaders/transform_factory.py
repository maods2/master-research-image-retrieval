import albumentations as A
from albumentations.pytorch import ToTensorV2

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(transform_config):
    """
    Factory function to get transformations based on the provided configuration.
    """
    transform_list = []

    if 'resize' in transform_config:
        resize_height, resize_width = transform_config['resize']
        transform_list.append(A.Resize(resize_height, resize_width))

    if transform_config.get('horizontal_flip', False):
        transform_list.append(A.HorizontalFlip(p=0.5))

    if transform_config.get('vertical_flip', False):
        transform_list.append(A.VerticalFlip(p=0.5))

    if 'rotation' in transform_config:
        transform_list.append(
            A.Rotate(
                limit=transform_config['rotation']['max_angle'],
                p=transform_config['rotation']['probability'],
            )
        )

    if 'color_jitter' in transform_config:
        transform_list.append(
            A.ColorJitter(
                brightness=transform_config['color_jitter']['brightness'],
                contrast=transform_config['color_jitter']['contrast'],
                saturation=transform_config['color_jitter']['saturation'],
                hue=transform_config['color_jitter']['hue'],
                p=transform_config['color_jitter']['probability'],
            )
        )

    if 'gaussian_noise' in transform_config:
        transform_list.append(
            A.GaussNoise(
                var_limit=tuple(
                    transform_config['gaussian_noise']['var_limit']
                ),
                mean=transform_config['gaussian_noise']['mean'],
                p=transform_config['gaussian_noise']['probability'],
            )
        )

    if 'gaussian_blur' in transform_config:
        transform_list.append(
            A.GaussianBlur(
                blur_limit=tuple(
                    transform_config['gaussian_blur']['blur_limit']
                ),
                p=transform_config['gaussian_blur']['probability'],
            )
        )

    if 'coarse_dropout' in transform_config:
        transform_list.append(
            A.CoarseDropout(
                max_holes=transform_config['coarse_dropout']['max_holes'],
                max_height=transform_config['coarse_dropout']['max_height'],
                max_width=transform_config['coarse_dropout']['max_width'],
                min_holes=transform_config['coarse_dropout']['min_holes'],
                min_height=transform_config['coarse_dropout']['min_height'],
                min_width=transform_config['coarse_dropout']['min_width'],
                fill_value=0,
                mask_fill_value=0,
                p=transform_config['coarse_dropout']['probability'],
            )
        )

    if 'distortion' in transform_config:
        transform_list.append(
            A.OneOf(
                [
                    A.OpticalDistortion(
                        p=transform_config['distortion']['optical_distortion']
                    ),
                    A.GridDistortion(
                        p=transform_config['distortion']['grid_distortion']
                    ),
                    A.PiecewiseAffine(
                        p=transform_config['distortion']['piecewise_affine']
                    ),
                ],
                p=transform_config['distortion']['probability'],
            )
        )

    if 'shift_scale_rotate' in transform_config:
        transform_list.append(
            A.ShiftScaleRotate(
                shift_limit=transform_config['shift_scale_rotate'][
                    'shift_limit'
                ],
                scale_limit=transform_config['shift_scale_rotate'][
                    'scale_limit'
                ],
                rotate_limit=transform_config['shift_scale_rotate'][
                    'rotate_limit'
                ],
                interpolation=1,
                border_mode=0,
                value=(0, 0, 0),
                p=transform_config['shift_scale_rotate']['probability'],
            )
        )

    if 'normalize' in transform_config:
        normalize_mean, normalize_std = tuple(
            transform_config['normalize']['mean']
        ), tuple(transform_config['normalize']['std'])
        transform_list.append(
            A.Normalize(mean=normalize_mean, std=normalize_std)
        )

    if transform_config.get('to_tensor', True):
        transform_list.append(ToTensorV2())

    return A.Compose(transform_list)
