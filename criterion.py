import torch
from torch import nn

def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        pred_words = pred.split()  # 予測された回答を単語に分割

        for answer in answers:
            num_match = 0
            answer_words = answer.split()  # 正解ラベルを単語に分割

            for pred_word in pred_words:
                if pred_word in answer_words:
                    num_match += 1

            acc += min(num_match / len(answer_words), 1)  # 正解ラベルの単語数で正規化して最大値を1に制限

        total_acc += acc / len(answers)  # 各正解ラベルに対する精度の平均を計算

    return total_acc / len(batch_pred)  # バッチ全体の平均精度を返す


def loss_fn(answers_ids, preds_logits, vocab_size):
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)

    answers_length = answers_ids.shape[1]
    preds_length = preds_logits.shape[1]

    if answers_length >= preds_length:
        answers_ids = answers_ids[:, :preds_length]
    else:
        preds_logits = preds_logits[:, :answers_length, :]

    answers_ids_flat = answers_ids.reshape(-1)
    preds_logits_flat = preds_logits.reshape(-1, vocab_size)

    return cross_entropy_loss(preds_logits_flat,  answers_ids_flat)