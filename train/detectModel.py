import numpy as np
import torch
import torch.nn as nn

from modeling_bert import BertForMaskedLM
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


class DetectModel(nn.Module):
    def __init__(self, tokenizer, embeddingsModel, classfierModel, device):
        super(DetectModel, self).__init__()
        self.tokenizer = tokenizer
        self.classifierModel = classfierModel
        self.embeddingsModel = embeddingsModel
        self.device = device

    # 获取梯度
    def get_gradients(self, input_token, bert_embedding, model, labels, attention_mask):
        # 获取句子的embedding

        embeddings = bert_embedding(input_token)
        embeddings.retain_grad()
        output = model(inputs_embeds=embeddings, token_type_ids=None, labels=labels,
                       attention_mask=attention_mask)
        res = output[0] if len(output) < 2 else output[1]
        label_index = torch.argmax(res, dim=1)
        # print('lable:0 for error and 1 for correct....', label_index)
        # print('正确' if label_index == 1 else '错误')
        # 分类模型预测的结果
        label = res[:, label_index][:, 0]
        # print('label:', label)
        # label=label.requires_grad_()
        # label = output[1][:, label_index]
        # label.backward(torch.tensor([1., 1., 1., 1.]), retain_graph=True)
        label.backward(torch.full(label.shape, 1.0).to(self.device), retain_graph=True)
        grads = embeddings.grad
        return grads, output

    def forward(self, input_token, labels, attention_mask=None):  # 主要任务是将上述网络连接起来
        # 获取bert的embedding层
        values = self.embeddingsModel.word_embeddings.weight.data
        pred_grads = []
        n = 1
        for i in range(n):
            # nlp任务中参照背景通常直接选零向量，所以这里
            # 让embedding层从零渐变到原始值，以实现路径变换。
            # alpha = 1.0 * i / (n - 1)
            # alpha = 0.9 + 0.001 * n
            # embeddings = values.to(device)
            # self.embeddingsModel.word_embeddings.weight.detach_()
            # print("Original_embedding: ", model.bert.embeddings.word_embeddings.weight)
            # self.embeddingsModel.word_embeddings.weight.data = values * alpha
            # self.embeddingsModel.word_embeddings.weight.requires_grad_(True)
            global output
            # 获取梯度
            pred_grad, output = self.get_gradients(input_token, self.embeddingsModel, self.classifierModel, labels,
                                                   attention_mask)

            pred_grad = pred_grad.cpu()
            pred_grads.append(pred_grad.numpy())

        pred_grads = np.mean(pred_grads, 0)

        # 这时候我们得到形状为(batch_size, seq_len, hidden_dim)的矩阵，我们要将它变换成(batch_size,seq_len)
        # 这时候有两种方案：1、直接求模长；2、取绝对值后再取最大。两者效果差不多。这里取第一种方案
        scores = np.sqrt((pred_grads ** 2).sum(axis=2))
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        scores = scores.round(4)
        input_token = input_token.cpu()
        results = []

        for i in range(input_token.shape[0]):
            sen = []
            for j in range(len(input_token[i])):
                t = input_token[i][j]
                s = scores[i][j]
                sen.append((self.tokenizer.decode([t]), s))
            results.append(sen)
        # results = [(self.tokenizer.decode([t]), s) for t, s in zip((input_token[i, :], scores[i, :]) for i in range(4))]
        # print(results[1:-1])
        # loss, logits = output[0], output[1]
        return scores, output


class IGBModel(nn.Module):
    def __init__(self, tokenizer, embeddingsModel, classifierModel, device, bertMaskedLM):
        super(IGBModel, self).__init__()
        self.tokenizer = tokenizer
        self.classifierModel = classifierModel
        self.embeddingsModel = embeddingsModel
        self.device = device
        self.detectModel = DetectModel(self.tokenizer, self.embeddingsModel, self.classifierModel, self.device)
        self.correctModel = bertMaskedLM
        bertMaskedLM.config.is_decoder = False

    def get_same_length_mask_e(self, scores_shape):
        t = torch.full(scores_shape, 103).long().to(self.device)
        return self.embeddingsModel(t)

    def forward(self, input_token, class_labels=None, mask_labels=None, attention_mask=None):  # 主要任务是将上述网络连接起来
        scores, output = self.detectModel(input_token, class_labels,
                                          attention_mask=attention_mask)  # 0 错误；1 正确
        if len(output) < 2:
            class_loss, logits = None, output[0]
        else:
            class_loss, logits = output[0], output[1]
        # print(scores)
        pre_class_labels = torch.argmax(logits, dim=1)

        # 若句子正确，则scores惩罚为0；否则保持不变
        scores_new = []
        for i in range(len(pre_class_labels)):
            if pre_class_labels[i].item() == 1:
                scores_new.append([0.0] * len(scores[i]))
            else:
                scores_new.append(scores[i])
        p = torch.tensor(scores_new,requires_grad=True).unsqueeze(dim=2).to(self.device)
        # p = torch.tensor([[data] for data in scores])

        # # 输出检查检测结果
        # p_out = p.squeeze(dim=2)[0].cpu()
        # input_id = input_token[0].cpu().tolist()
        # results = [(self.tokenizer.decode([t]), s.item()) for t, s in zip(input_id, p_out)]
        # # print(''.join([self.tokenizer.decode([id]) for id in input_id[1:-1]]))
        # label = pre_class_labels[0].item()
        # print('预测结果:' + ('句子正确' if label == 1 else '句子错误'))
        # print(results[1:-1])

        mask_e = self.get_same_length_mask_e(scores.shape).to(self.device)
        e = self.embeddingsModel(input_token).to(self.device)
        # 将mask_e和原始e加权求和
        e_ = p * mask_e + (1 - p) * e
        correct_outputs = self.correctModel(inputs_embeds=e_, return_dict=True, labels=mask_labels,
                                            attention_mask=attention_mask)
        # correct_loss, correct_logits = correct_outputs[0], correct_outputs[1]
        # prediction_scores = logits
        # pre = torch.softmax(prediction_scores, -1)
        # top_info = torch.topk(pre, k=5)
        # scores = top_info[0]
        # ids = top_info[1]
        # for line in range(ids.size()[0]):
        #     for i in range(20):
        #         predicted_token = self.tokenizer.convert_ids_to_tokens(ids[line][i])
        #         print(predicted_token)
        #     print('*' * 50)
        if len(correct_outputs) < 2:
            correct_loss, logits = None, correct_outputs[0]
        else:
            correct_loss, logits = correct_outputs[0], correct_outputs[1]

        # if class_loss is not None:
        #     sum_loss = class_loss + correct_loss
        # else:
        #     sum_loss = None
        # print('class_loss:', class_loss)
        # print('correct_loss:', correct_loss)
        # print('a')
        return correct_loss, logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    tokenizer = BertTokenizer.from_pretrained('../../work/model/bert-base-chinese-class-finetuned')
    # bert = BertModel.from_pretrained('../../work/model/bert-base-chinese-class-finetuned')
    bertMaskedLM = BertForMaskedLM.from_pretrained('../../work/model/bert-base-chinese-finetuned')

    classifierModel = BertForSequenceClassification.from_pretrained(
        "../../work/model/bert-base-chinese-class-finetuned",
        num_labels=2)  # num_labels表示2个分类，错误和正确
    # 获取bert的embedding层
    embeddingsModel = classifierModel.bert.embeddings
    iGBModel = IGBModel(tokenizer, embeddingsModel, classifierModel, device, bertMaskedLM)
    iGBModel.to(device)
    text = '我今天很高行'
    input_id = tokenizer.encode(text)
    input_token = torch.tensor([input_id]).long().to(device)
    # 分类标签
    class_labels = torch.tensor([[1]]).to(device)
    # 掩码标签
    text_label = '我今天很高兴'
    label_id = tokenizer.encode(text_label)
    mask_labels = torch.tensor([label_id]).long().to(device)

    outputs = iGBModel(input_token, class_labels, mask_labels, attention_mask=None)[1]

    # print(scores.tolist())
    prediction_scores = outputs
    pre = torch.softmax(prediction_scores[0], -1)
    top_info = torch.topk(pre, k=5)
    scores = top_info[0]
    ids = top_info[1]
    for i in range(ids.size()[0]):
        predicted_token = tokenizer.convert_ids_to_tokens(ids[i])
        print(predicted_token)
    print('a')
