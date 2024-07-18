import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ViltForQuestionAnswering
import torch.nn as nn
from transformers import AdamW, get_scheduler
from tqdm import tqdm
from datasets import train_loader, vilt_config
from tqdm import tqdm
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def format_float(value):
    # floatを文字列に変換
    value_str = str(value)

    # 小数点をアンダースコアに置換
    formatted_value = value_str.replace('.', '_')

    # 上から10桁までで切り捨てる
    if len(formatted_value) > 10:
        formatted_value = formatted_value[:10]

    return formatted_value

cache_dir = "chache_transformers"
os.environ['TRANSFORMERS_CACHE'] = cache_dir

model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 id2label=vilt_config.id2label,
                                                 label2id=vilt_config.label2id,
                                                 cache_dir=cache_dir)


# Define optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 50
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
model.to(device)
train_losses = []

model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for batch in tqdm(train_loader, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss+=loss.item()

    torch.save(model.state_dict(), f"models/model_{epoch}_{format_float(train_loss)}.pth")
    train_losses.append(train_loss)

    print(f"【{epoch + 1}/{num_epochs}")
    print(f"train loss: {train_losses[epoch]:.4f}\n")