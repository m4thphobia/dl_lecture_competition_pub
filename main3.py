import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import detect_anomaly
from dataloader import train_loader, test_loader
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

from networks import MLPAdapter, VQAModel
from criterion import loss_fn
from tqdm import tqdm, trange
import time
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, dataloader, optimizer, loss_fn, device):
    model.to(device)
    model.train()

    total_loss = 0
    total_acc = 0
    simple_acc = 0

    start = time.time()
    for images, questions, answers in tqdm(dataloader, leave=False):
        images =  images.to(device)
        questions_ids = llm_tokenizer(questions, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)
        answers_ids = llm_tokenizer(answers, padding=True, truncation=True, return_tensors="pt")["input_ids"][:, 1:].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            preds_logits = model(images, questions_ids)
            loss = loss_fn(answers_ids, preds_logits, vocab_size)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        torch.cuda.empty_cache()
        del images, questions_ids, answers_ids, preds_logits, loss
        # total_acc += VQA_criterion(outputs, answers)  # VQA accuracy

    return total_loss / len(dataloader), time.time() - start #, , total_acc / len(dataloader)


def plot_loss(train_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(out/train_loss.png)

def format_float(value):
    # floatを文字列に変換
    value_str = str(value)

    # 小数点をアンダースコアに置換
    formatted_value = value_str.replace('.', '_')

    # 上から10桁までで切り捨てる
    if len(formatted_value) > 10:
        formatted_value = formatted_value[:10]

    return formatted_value

if __name__ == "__main__":

    cache_dir = "chache_transformers"
    os.environ['TRANSFORMERS_CACHE'] = cache_dir

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)

    # Vicuna-7Bモデルのロード
    llm_model = AutoModelForCausalLM.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        quantization_config = BitsAndBytesConfig(load_in_4bit=True),
        torch_dtype=torch.float32,
        device_map=device,
        cache_dir=cache_dir,
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(
        "lmsys/vicuna-7b-v1.5",
        use_fast=False,
        cache_dir=cache_dir,
    )
    vocab_size = len(llm_tokenizer)

    # MLPアダプタの初期化
    input_dim = 768  # CLIPの出力次元
    output_dim = llm_model.config.hidden_size  # Vicunaのトークン埋め込み次元
    adapter = MLPAdapter(input_dim, output_dim)

    vqa_model = VQAModel(clip_model, clip_processor, adapter, llm_model, llm_tokenizer)

    num_epoch = 500
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=65536)
    train_losses = []

    for epoch in trange(num_epoch, leave=False):
        train_loss, train_time = train(vqa_model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)

        if (epoch) % 3 == 0:
            torch.save(vqa_model.state_dict(), f"models/model_{epoch}_{format_float(train_loss)}.pth")
            #plot_loss(train_losses)

        print(f"【{epoch + 1}/{num_epoch}")
        print(f"train time: {train_time:.2f} [s]")
        print(f"train loss: {train_loss:.4f}\n")

    # 提出用ファイルの作成
    vqa_modelmodel.eval()
    submission = []
    for image, question in test_loader:
        images =  images.to(device)
        questions_ids = llm_tokenizer(questions, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)

        pred = vqa_model.generate_answer(image, question)
        submission.append(pred)

    submission = np.array(submission)
    torch.save(vqa_modelmodel.state_dict(), "_model.pth")
    np.save("_submission.npy", submission)