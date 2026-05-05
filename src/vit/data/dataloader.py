from typing import Dict, Any, Tuple, Type
from torch.utils.data import DataLoader, Dataset

from .datasets import EmotionDataset, BinaryEmotionDataset
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms


def _dataset_class(config: Dict[str, Any]) -> Type[Dataset]:
    dataset_type = config['data'].get('dataset_type', 'standard')
    if dataset_type == 'binary':
        return BinaryEmotionDataset
    return EmotionDataset


def _dataset_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    image_path_column = config['data'].get('image_path_column')
    if image_path_column is not None:
        kwargs['image_path_column'] = image_path_column
    return kwargs


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    test_transform = get_test_transforms(config)

    DatasetClass = _dataset_class(config)
    extra = _dataset_kwargs(config)

    train_dataset = DatasetClass(
        csv_file=config['data']['train_csv'],
        img_dir=config['data']['train_img_dir'],
        transform=train_transform,
        **extra
    )
    
    val_dataset = DatasetClass(
        csv_file=config['data']['val_csv'],
        img_dir=config['data']['val_img_dir'],
        transform=val_transform,
        **extra
    )
    
    test_dataset = DatasetClass(
        csv_file=config['data']['test_csv'],
        img_dir=config['data']['test_img_dir'],
        transform=test_transform,
        **extra
    )
    
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    pin_memory = config['training']['pin_memory']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader

def get_dataset_info(config: Dict[str, Any]) -> Dict[str, Any]:
    DatasetClass = _dataset_class(config)
    extra = _dataset_kwargs(config)

    train_dataset = DatasetClass(
        csv_file=config['data']['train_csv'],
        img_dir=config['data']['train_img_dir'],
        transform=get_train_transforms(config),
        **extra
    )
    
    val_dataset = DatasetClass(
        csv_file=config['data']['val_csv'],
        img_dir=config['data']['val_img_dir'],
        transform=get_val_transforms(config),
        **extra
    )
    
    test_dataset = DatasetClass(
        csv_file=config['data']['test_csv'],
        img_dir=config['data']['test_img_dir'],
        transform=get_test_transforms(config),
        **extra
    )
    
    return {
        'num_emotions': train_dataset.get_num_emotions(),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'emotion_columns': train_dataset.EMOTION_COLUMNS
    }
