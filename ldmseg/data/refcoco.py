import os
import json
import torch

import numpy as np
import torch.utils.data as data
from PIL import Image
from typing import Optional, Any, Tuple
import random
from collections import defaultdict

from ldmseg.data.util.mypath import MyPath
from ldmseg.utils.utils import color_map
from ldmseg.data.util.mask_generator import MaskingGenerator

from ldmseg.data.refer.refer import REFER



class ReferDataset(data.Dataset):
    def __init__(
        self,
        prefix: str = "./refer/data",
        split: str = 'val',
        tokenizer: Optional[Any] = None,
        transform = None,
        download: bool = False,
        remap_labels: bool = False,
        caption_dropout: float = 0.0,
        overfit: bool = False,
        encoding_mode: str = 'color',
        caption_type: str = 'none',
        inpaint_mask_size: Optional[Tuple[int]] = None,
        num_classes: int = 128,
        fill_value: int = 0.5,
        ignore_label: int = 0,
        inpainting_strength: float = 0.0,
        refcoco_name='refcoco',
        refcoco_split='val'
    ):
        self.REFCOCO_CATEGORIES = [
            {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "instance"},
            {"color": [119, 11, 32], "isthing": 1, "id": 0, "name": "background"}
        ]
        self.REFCOCO_CATEGORY_NAMES = [k["name"] for k in self.REFCOCO_CATEGORIES]
        
        self.classes = []
        self.transform = transform
        self.split = split
        self.refer = REFER(prefix, 'refcoco', 'unc')

        self.max_tokens = 77
        self.ignore_label = -1
        ref_ids = self.refer.getRefIds(split=self.split)
        img_ids = self.refer.getImgIds(ref_ids)

        all_imgs = self.refer.Imgs
        self.imgs = list(all_imgs[i] for i in img_ids)
        self.ref_ids = ref_ids
        # record used referrings(values) for the image(key) when eval
        self.ref_ids_cur_ptr = {}
        self.input_ids = []
        self.attention_masks = []
        self.input_ids_flip = []
        self.raw_sentences = []

        self.tokenizer = tokenizer
        
        self.eval_mode = False if split=='train' else True 
        # if we are testing on a dataset, test all sentences of an object;
        # o/w, we are validating during training, randomly sample one sentence for efficiency
        for r in ref_ids:
            ref = self.refer.Refs[r]

            sentences_for_ref = []
            attentions_for_ref = []
            raw_sen_for_ref = []

            for i, (el, sent_id) in enumerate(zip(ref['sentences'], ref['sent_ids'])):
                sentence_raw = el['sent']

                input_ids = self.tokenizer(text=sentence_raw, truncation=True, max_length=self.max_tokens, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
                
                # truncation of tokens
                padded_input_ids = input_ids['input_ids'][:, :self.max_tokens]
                attention_mask = input_ids['attention_mask'][:, :self.max_tokens]
                
                sentences_for_ref.append(padded_input_ids)
                attentions_for_ref.append(attention_mask)
                raw_sen_for_ref.append(sentence_raw)
            
            
            self.input_ids.append(sentences_for_ref)
            self.attention_masks.append(attentions_for_ref)
            self.raw_sentences.append(raw_sen_for_ref)

    def get_classes(self):
        return self.classes
    
    def get_class_names(self):
        return self.REFCOCO_CATEGORY_NAMES
    
    def __len__(self):
        return len(self.ref_ids)

    def _load_semseg(self, index, mode='array'):
        _semseg = np.array(Image.open(self.semsegs[index]).convert('RGB'))
        _semseg = _semseg[:, :, 0] + 256 * _semseg[:, :, 1] + (256 ** 2) * _semseg[:, :, 2]

        # count pixels for each unique instance and set to ignore if below pixel threshold
        small_instances = set()
        if self.training:
            if self.pixel_threshold > 0:
                ids, counts = np.unique(_semseg, return_counts=True)
                for i, c in zip(ids, counts):
                    if c < self.pixel_threshold:
                        _semseg[_semseg == i] = self.ignore_label
                        small_instances.add(i)

        # load segments info
        key = self.semsegs[index].split('/')[-1]
        segments_info = self.annotations_dict[key]['segments_info']
        keep_segments_info = {}
        for seg in segments_info:
            if seg['id'] in small_instances:
                continue
            if seg['iscrowd'] and self.training:
                _semseg[_semseg == seg['id']] = self.ignore_label
                continue

            assert seg['iscrowd'] == 0 or not self.training
            keep_segments_info[seg['id']] = {'category_id': seg['category_id'],
                                             'iscrowd': seg['iscrowd'],
                                             'category_name': self.cat_info[seg['category_id']]['name'],
                                             'isthing': self.cat_info[seg['category_id']]['isthing']
                                             }
            # remap category ids to contiguous ids
            curr_cat_id = keep_segments_info[seg['id']]['category_id']
            if curr_cat_id in self.meta_data["thing_dataset_id_to_contiguous_id"]:
                keep_segments_info[seg['id']]['category_id'] = self.meta_data["thing_dataset_id_to_contiguous_id"][curr_cat_id]  # noqa
            else:
                keep_segments_info[seg['id']]['category_id'] = self.meta_data["stuff_dataset_id_to_contiguous_id"][curr_cat_id]  # noqa
            assert keep_segments_info[seg['id']]['category_id'] < 133  # 133 is the number of classes in COCO panoptic

        # load captions
        image_id = key.split('.')[0]
        captions_info = self.captions_dict[int(image_id)]

        # assert
        assert _semseg.max() > 0
        assert len(keep_segments_info) == len([x for x in np.unique(_semseg) if x != self.ignore_label])

        if mode == 'pil':
            return Image.fromarray(_semseg.astype(np.uint8))

        return _semseg, keep_segments_info, captions_info, image_id + '.jpg'

    def __getitem__(self, index):
        sample = {}
        this_ref_id = self.ref_ids[index]
        this_img_id = self.refer.getImgIds(this_ref_id)
        this_img = self.refer.Imgs[this_img_id[0]]

        img = Image.open(os.path.join(self.refer.IMAGE_DIR, this_img['file_name'])).convert("RGB")
        sample['image'] = img
        
        ref = self.refer.loadRefs(this_ref_id)

        ref_mask = np.array(self.refer.getMask(ref[0])['mask'])
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 1] = 1
        cat_name = self.refer.Cats[ref[0]['category_id']]

        annot = Image.fromarray(annot.astype(np.uint8))
        sample['semseg'] = annot
        sample['mask'] = np.ones_like(annot)
        sample['mask'] = Image.fromarray(sample['mask'])
        sample['cat'] = cat_name

        # meta data
        sample['meta'] = {
            'im_size': (img.size[1], img.size[0]),
            'image_file': os.path.join(self.refer.IMAGE_DIR, this_img['file_name']),
            "image_id": this_img_id,
            'segments_info': None,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        
        sample['image_semseg'] = sample['semseg'].unsqueeze(0)
        sample['inpainting_mask'] = torch.randint(0, 2, (64, 64), dtype=torch.int).bool()
        if self.eval_mode:
            assert len(self.raw_sentences[index]) == len(self.input_ids[index])
            ref_nums = len(self.raw_sentences[index])
            if index not in self.ref_ids_cur_ptr:
                # images not evaled
                self.ref_ids_cur_ptr[index] = 0
            assert self.ref_ids_cur_ptr[index] < ref_nums
            tensor_embeddings = self.input_ids[index][self.ref_ids_cur_ptr[index]]
            raw_sentences = self.raw_sentences[index][self.ref_ids_cur_ptr[index]]
            self.ref_ids_cur_ptr[index] += 1
            sample['tokens'] = tensor_embeddings
            sample['text'] = raw_sentences
        else:
            choice_sent = np.random.choice(len(self.input_ids[index]))
            tensor_embeddings = self.input_ids[index][choice_sent]
            attention_mask = self.attention_masks[index][choice_sent]
            sample['tokens'] = tensor_embeddings.squeeze(0)
            sample['text'] = self.raw_sentences[index][choice_sent]
        return sample