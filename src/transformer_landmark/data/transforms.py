from typing import Dict, Any, List
from torchvision import transforms


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    image_size = config['data']['image_size']
    aug_config = config['augmentation']
    
    transform_list: List[Any] = [
        transforms.Resize((image_size, image_size)),
    ]
    
    if aug_config['train']['random_horizontal_flip']:
        flip_prob = aug_config['train']['flip_prob']
        transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
    
    if aug_config['train'].get('random_rotation', False):
        rotation_degrees = aug_config['train']['rotation_degrees']
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))
    
    if aug_config['train'].get('random_affine', False):
        affine_degrees = aug_config['train']['affine_degrees']
        affine_scale = tuple(aug_config['train']['affine_scale'])
        transform_list.append(transforms.RandomAffine(
            degrees=affine_degrees,
            scale=affine_scale
        ))
    
    if aug_config['train'].get('random_perspective', False):
        distortion = aug_config['train']['perspective_distortion']
        transform_list.append(transforms.RandomPerspective(
            distortion_scale=distortion,
            p=0.5
        ))
    
    if aug_config['train']['color_jitter']:
        transform_list.append(transforms.ColorJitter(
            brightness=aug_config['train']['brightness'],
            contrast=aug_config['train']['contrast'],
            saturation=aug_config['train']['saturation'],
            hue=aug_config['train']['hue']
        ))
    
    if aug_config['train'].get('random_grayscale', False):
        grayscale_prob = aug_config['train']['grayscale_prob']
        transform_list.append(transforms.RandomGrayscale(p=grayscale_prob))
    
    transform_list.append(transforms.ToTensor())
    
    if aug_config['train'].get('random_erasing', False):
        erasing_prob = aug_config['train']['erasing_prob']
        transform_list.append(transforms.RandomErasing(p=erasing_prob))
    
    transform_list.append(transforms.Normalize(
        mean=aug_config['imagenet_norm']['mean'],
        std=aug_config['imagenet_norm']['std']
    ))
    
    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    image_size = config['data']['image_size']
    aug_config = config['augmentation']
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config['imagenet_norm']['mean'],
            std=aug_config['imagenet_norm']['std']
        )
    ])


def get_test_transforms(config: Dict[str, Any]) -> transforms.Compose:
    return get_val_transforms(config)
