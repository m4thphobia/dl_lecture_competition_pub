import os
from PIL import Image
import random
import re
import pandas
from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from transformers import ViltProcessor
from transformers import ViltConfig

cache_dir = "chache_transformers"
os.environ['TRANSFORMERS_CACHE'] = cache_dir

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class VQADataset(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, processor, config, transform=None, answer=True):
        self.df = pandas.read_json(df_path)
        self.image_dir = image_dir
        self.processor = processor
        self.config = config
        self.transform = transform
        self.answer = answer
        if self.answer:
            self._make_annotation()

    def _make_annotation(self):
        self.annotations = []
        dict = {}
        for answer_sets in self.df["answers"]:
            for i in range(len(answer_sets)):
                answer_sets[i]['answer'] = process_text(answer_sets[i]['answer'])
            dict['answers'] = answer_sets
            self.annotations.append(dict)

        for annotation in self.annotations:
            answers = annotation['answers']
            answer_count = {}
            for answer in answers:
                answer_ = answer["answer"]
                answer_count[answer_] = answer_count.get(answer_, 0) + 1
            labels = []
            scores = []
            for answer in answer_count:
                if answer not in list(self.config.label2id.keys()):
                    continue
                labels.append(self.config.label2id[answer])
                score = self._get_score(answer_count[answer])
                scores.append(score)
            annotation['labels'] = labels
            annotation['scores'] = scores

    def _get_score(self, count: int) -> float:
        return min(1.0, count / 3)

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = self.df["question"][idx]
        question = process_text(question)

        if self.answer:
            annotation = self.annotations[idx]
            encoding = self.processor(image, question, padding="max_length", truncation=True, return_tensors="pt")
            # remove batch dimension
            for k,v in encoding.items():
                encoding[k] = v.squeeze()
            # add labels
            labels = annotation['labels']
            scores = annotation['scores']
            # based on: https://github.com/dandelin/ViLT/blob/762fd3975c180db6fc88f577cf39549983fa373a/vilt/modules/objectives.py#L301
            targets = torch.zeros(len(self.config.id2label))
            for label, score in zip(labels, scores):
                targets[label] = score
            encoding["labels"] = targets

            return encoding
        else:
            encoding = self.processor(images=image, text=process_text(question), padding='max_length', truncation=True, return_tensors="pt")
            for k,v in encoding.items():
                encoding[k] = v.squeeze()
            return encoding

    def __len__(self):
        return len(self.df)

# Transform for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
vilt_processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm', cache_dir=cache_dir)
vilt_config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa", cache_dir=cache_dir)

train_dataset = VQADataset(df_path="../data/train.json", image_dir="../data/train", processor=vilt_processor, config=vilt_config, transform=transform)
test_dataset = VQADataset(df_path="../data/valid.json", image_dir="../data/valid", processor=vilt_processor, config=vilt_config, transform=transform, answer=False)


print(train_dataset[0].keys())

def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  pixel_values = [item['pixel_values'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  token_type_ids = [item['token_type_ids'] for item in batch]
  labels = [item['labels'] for item in batch]

  # create padded pixel values and corresponding pixel mask
  encoding = vilt_processor.image_processor.pad(pixel_values, return_tensors="pt")

  # create new batch
  batch = {}
  batch['input_ids'] = torch.stack(input_ids)
  batch['attention_mask'] = torch.stack(attention_mask)
  batch['token_type_ids'] = torch.stack(token_type_ids)
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = torch.stack(labels)

  return batch

train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=16, shuffle=True)