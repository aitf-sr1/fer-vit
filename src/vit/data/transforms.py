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
    
    if aug_config['train']['color_jitter']:
        transform_list.append(transforms.ColorJitter(
            brightness=aug_config['train']['brightness'],
            contrast=aug_config['train']['contrast'],
            saturation=aug_config['train']['saturation'],
            hue=aug_config['train']['hue']
        ))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug_config['imagenet_norm']['mean'],
            std=aug_config['imagenet_norm']['std']
        )
    ])
    
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
