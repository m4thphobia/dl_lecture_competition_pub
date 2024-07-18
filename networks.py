import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPAdapter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        return x

class VQAModel(nn.Module):
    def __init__(self, clip_model, clip_processor, adapter, llm_model, llm_tokenizer):
        super().__init__()
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.adapter = adapter
        self.llm_model = llm_model
        self.llm_tokenizer = llm_tokenizer

    def generate_answer(self, image, questions_ids):

        _image = self.clip_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        image_feature = self.clip_model.get_image_features(_image)
        adapted_feature = self.adapter(image_feature)
        adapted_feature = adapted_feature.unsqueeze(1)

        #_question = self.llm_tokenizer(question, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        question_embeddings = self.llm_model.get_input_embeddings()(questions_ids)

        concatenated_feature = torch.cat((adapted_features, question_embeddings), dim=1)
        output = self.llm_model(inputs_embeds=concatenated_feature)
        output_ids = torch.argmax(output.logits, dim=-1)

        output_text = self.llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    def forward(self, images, questions_ids):
        if images.dim() == 3:
            images = images.unsqueeze(0)

        _images = self.clip_processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        image_features = self.clip_model.get_image_features(_images)
        adapted_features = self.adapter(image_features)
        adapted_features = adapted_features.unsqueeze(1)

        # _questions = self.llm_tokenizer(questions, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        question_embeddings = self.llm_model.get_input_embeddings()(questions_ids)

        concatenated_features = torch.cat((adapted_features, question_embeddings), dim=1)
        outputs = self.llm_model(inputs_embeds=concatenated_features)
        return outputs.logits