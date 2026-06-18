from typing import Dict, Any, Tuple, Type
from torch.utils.data import DataLoader, Dataset

from .datasets import EmotionDataset, BinaryEmotionDataset
from .multimodal_dataset import MultiModalEmotionDataset
from .parquet_dataset import ParquetEmotionDataset
from .transforms import get_train_transforms, get_val_transforms, get_test_transforms

_MULTIMODAL_TYPES = {
    'multimodal_landmarks': 'mediapipe_landmarks',
    'multimodal_au': 'action_units',
    'multimodal_both': 'both',
}


def _dataset_class(config: Dict[str, Any]) -> Type[Dataset]:
    dataset_type = config['data'].get('dataset_type', 'standard')
    if dataset_type == 'binary':
        return BinaryEmotionDataset
    if dataset_type == 'parquet':
        return ParquetEmotionDataset
    if dataset_type in _MULTIMODAL_TYPES:
        return MultiModalEmotionDataset
    return EmotionDataset


def _dataset_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    dataset_type = config['data'].get('dataset_type', 'standard')
    kwargs: Dict[str, Any] = {}

    image_path_column = config['data'].get('image_path_column')
    if image_path_column is not None:
        kwargs['image_path_column'] = image_path_column

    if dataset_type == 'parquet':
        kwargs['filter_synthetic'] = config['data'].get('filter_synthetic', False)
        return kwargs

    if dataset_type in _MULTIMODAL_TYPES:
        kwargs['aux_type'] = _MULTIMODAL_TYPES[dataset_type]
        kwargs['au_columns'] = config['data'].get('au_columns', [])

    return kwargs


def _dataset_transform(config: Dict[str, Any], split: str) -> Any:
    if split == 'train':
        return get_train_transforms(config)
    if split == 'val':
        return get_val_transforms(config)
    return get_test_transforms(config)


def _build_dataset_for_split(
    config: Dict[str, Any],
    split: str,
) -> Dataset:
    dataset_type = config['data'].get('dataset_type', 'standard')
    DatasetClass = _dataset_class(config)
    extra = _dataset_kwargs(config)
    transform = _dataset_transform(config, split)

    if dataset_type == 'parquet':
        parquet_key = f'{split}_parquet'
        return DatasetClass(
            parquet_path=config['data'][parquet_key],
            transform=transform,
            **extra,
        )

    return DatasetClass(
        csv_file=config['data'][f'{split}_csv'],
        img_dir=config['data'][f'{split}_img_dir'],
        transform=transform,
        **extra,
        **_aux_csv_for_split(config, split),
    )


def _aux_csv_for_split(config: Dict[str, Any], split: str) -> Dict[str, Any]:
    dataset_type = config['data'].get('dataset_type', 'standard')
    if dataset_type not in _MULTIMODAL_TYPES:
        return {}
    key = f'{split}_aux_csv'
    if key in config['data']:
        return {'aux_csv': config['data'][key]}
    if 'aux_csv' in config['data']:
        return {'aux_csv': config['data']['aux_csv']}
    return {}


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = _build_dataset_for_split(config, 'train')
    val_dataset = _build_dataset_for_split(config, 'val')
    test_dataset = _build_dataset_for_split(config, 'test')

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
        num_workers=config['training'].get('val_num_workers', num_workers),
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training'].get('test_num_workers', num_workers),
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader

def get_dataset_info(config: Dict[str, Any]) -> Dict[str, Any]:
    train_dataset = _build_dataset_for_split(config, 'train')
    val_dataset = _build_dataset_for_split(config, 'val')
    test_dataset = _build_dataset_for_split(config, 'test')

    return {
        'num_emotions': train_dataset.get_num_emotions(),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
        'emotion_columns': train_dataset.EMOTION_COLUMNS
    }
