import torch
from torch import nn
from torchvision import transforms as T
from typing import Callable, Dict, Tuple, Any, Optional


class DatasetBase(object):

    def __init__(
        self,
        data_dir: str
    ) -> None:
        """ Base class for datasets
        """

        self.data_dir = data_dir

    def get_train_transforms(
        self,
        p: Dict[str, Any]
    ) -> Callable:
        """ Returns a composition of transformations to be applied to the training images
        """

        normalize = T.Normalize(**p['normalize_params']) if p['normalize'] else nn.Identity()
        if p['type'] == 'resize_pil':
            from .util import pil_transforms as pil_tr

            size = p['size']
            normalize = pil_tr.Normalize(**p['normalize_params']) if p['normalize'] else nn.Identity()
            transforms = T.Compose([
                # pil_tr.RandomHorizontalFlip() if p['flip'] else nn.Identity(),
                pil_tr.Resize((size, size)),
                pil_tr.ToTensor(),
                normalize
            ])

        else:
            raise NotImplementedError(f'Unknown transformation type {p["type"]}')

        return transforms

    def get_val_transforms(
        self,
        p: Dict
    ) -> Callable:
        """ Returns a composition of transformations to be applied to the validation images
        """

        normalize = T.Normalize(**p['normalize_params']) if p['normalize'] else nn.Identity()
        if p['type'] in ['resize_pil', 'random_resize_pil']:
            from .util import pil_transforms as pil_tr
            size = p['size']
            normalize = pil_tr.Normalize(**p['normalize_params']) if p['normalize'] else nn.Identity()
            transforms = T.Compose([
                pil_tr.Resize((size, size)),
                pil_tr.ToTensor(),
                normalize
            ])

        else:
            raise NotImplementedError(f'Unknown transformation type {p["type"]}')

        return transforms

    def get_dataset(
        self,
        db_name: str,
        *,
        split: Any,
        tokenizer: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        remap_labels: bool = False,
        caption_dropout: float = 0.0,
        download: bool = False,
        overfit: bool = False,
        encoding_mode: str = 'color',
        caption_type: Optional[str] = 'none',
        inpaint_mask_size: Optional[Tuple[int]] = None,
        num_classes: Optional[int] = None,
        fill_value: Optional[int] = None,
        ignore_label: Optional[int] = None,
        inpainting_strength: Optional[float] = None,
        refcoco_split='val'
    ) -> Any:
        """ Returns the dataset to be used for training or evaluation
        """

        if db_name in 'coco':
            from .coco import COCO
            dataset_cls = COCO
        elif db_name in ['refcoco', 'refcoco+', 'refcocog']:
            from .refcoco import ReferDataset
            dataset_cls = ReferDataset
        elif db_name == ['refcoco_eval', 'refcoco+_eval', 'refcocog_eval']:
            from .refcoco_eval import ReferDataset_Eval
            dataset_cls = ReferDataset_Eval
        else:
            raise NotImplementedError(f'Unknown dataset {db_name}')

        if isinstance(split, list):
            datasets = [
                dataset_cls(
                    prefix=self.data_dir,
                    split=sp,
                    transform=transform,
                    download=download,
                    remap_labels=remap_labels,
                    tokenizer=tokenizer,
                    caption_dropout=caption_dropout,
                    overfit=overfit,
                    caption_type=caption_type,
                    encoding_mode=encoding_mode,
                    inpaint_mask_size=inpaint_mask_size,
                    num_classes=num_classes,
                    fill_value=fill_value,
                    ignore_label=ignore_label,
                    inpainting_strength=inpainting_strength,
                ) for sp in split
                ]
            return torch.utils.data.ConcatDataset(datasets)
        else:
            dataset = dataset_cls(
                prefix=self.data_dir,
                split=split,
                transform=transform,
                download=download,
                remap_labels=remap_labels,
                tokenizer=tokenizer,
                caption_dropout=caption_dropout,
                overfit=overfit,
                caption_type=caption_type,
                encoding_mode=encoding_mode,
                inpaint_mask_size=inpaint_mask_size,
                num_classes=num_classes,
                fill_value=fill_value,
                ignore_label=ignore_label,
                inpainting_strength=inpainting_strength,
                refcoco_name=db_name,
                refcoco_split=refcoco_split
            )
            return dataset
