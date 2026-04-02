from typing import Dict, Any, Tuple
import pandas as pd
from torch.utils.data import DataLoader

from .datasets import LandmarkEmotionDataset
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)
    test_transform = get_test_transforms(config)
    
    mode = config.get('model', {}).get('mode', 'classification')
    
    train_dataset = LandmarkEmotionDataset(
        csv_file=config['data']['train_csv'],
        transform=train_transform,
        mode=mode
    )
    
    val_dataset = LandmarkEmotionDataset(
        csv_file=config['data']['val_csv'],
        transform=val_transform,
        mode=mode
    )
    
    test_dataset = LandmarkEmotionDataset(
        csv_file=config['data']['test_csv'],
        transform=test_transform,
        mode=mode
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


def get_emotion_statistics(config: Dict[str, Any]) -> pd.DataFrame:
    train_dataset = LandmarkEmotionDataset(
        csv_file=config['data']['train_csv'],
        transform=get_train_transforms(config)
    )
    
    return train_dataset.get_emotion_statistics()


def get_dataset_info(config: Dict[str, Any]) -> Dict[str, Any]:
    train_dataset = LandmarkEmotionDataset(
        csv_file=config['data']['train_csv'],
        transform=get_train_transforms(config)
    )
    
    val_dataset = LandmarkEmotionDataset(
        csv_file=config['data']['val_csv'],
        transform=get_val_transforms(config)
    )
    
    test_dataset = LandmarkEmotionDataset(
        csv_file=config['data']['test_csv'],
        transform=get_test_transforms(config)
    )
    
    return {
        'num_emotions': train_dataset.get_num_emotions(),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'emotion_columns': train_dataset.EMOTION_COLUMNS
    }
