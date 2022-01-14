import random
import sys
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
sys.path.append('../')
from work.modeling_bert import BertForSequenceClassification
from work.optimization import get_linear_schedule_with_warmup, AdamW
from work.tokenization_bert import BertTokenizer


# 设定超参数
SEED = 123
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 读取文件
def readfile(filename):
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
        return content


pos_text, neg_text = readfile('../data/classifier_train_data/sig_all_correct_train.txt')[0:], readfile('../data/classifier_train_data/sig_all_mistake_train.txt')[0:]
sentences = pos_text + neg_text

# 设定标签
pos_targets = np.ones((len(pos_text)))  # 1:代表正确
neg_targets = np.zeros((len(neg_text)))  # 0:代表错误
targets = np.concatenate((pos_targets, neg_targets), axis=0).reshape(-1, 1)  # (10000, 1)
total_targets = torch.tensor(targets)

# 对句子进行编码
tokenizer = BertTokenizer.from_pretrained('../model/bert-base-chinese')
print(pos_text[2])
print(tokenizer.tokenize(pos_text[2]))
print(tokenizer.encode(pos_text[2]))
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(pos_text[2])))


# 将每一句转成数字（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
def convert_text_to_token(tokenizer, sentence, limit_size=126):
    tokens = tokenizer.encode(sentence[:limit_size])  # 直接截断
    if len(tokens) < limit_size + 2:  # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


input_ids = [convert_text_to_token(tokenizer, sen) for sen in sentences]

input_tokens = torch.tensor(input_ids)
print(input_tokens.shape)  # torch.Size([10000, 128])


# attention_masks,在一个文本中，如果是pad符号则是0，否则就是1
# 建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks

atten_masks = attention_masks(input_ids)
attention_tokens = torch.tensor(atten_masks)

# 划分训练集和测试集
from sklearn.model_selection import train_test_split

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_tokens, total_targets, random_state=666,
                                                                        test_size=0.1)
train_masks, test_masks, _, _ = train_test_split(attention_tokens, input_tokens, random_state=666, test_size=0.1)
print(train_inputs.shape, test_inputs.shape)  # torch.Size([8000, 128]) torch.Size([2000, 128])
print(train_masks.shape)  # torch.Size([8000, 128])和train_inputs形状一样

# 创建DataLoader,用来取出一个batch的数据
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# 创建模型，优化器
model = BertForSequenceClassification.from_pretrained("../model/bert-base-chinese",
                                                      num_labels=2)  # num_labels表示2个分类，好评和差评

# device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPSILON)

# 学习率预热，训练时先从小的学习率开始训练
epochs = 15
# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# 训练。评估模型
def binary_acc(preds, labels):  # preds.shape=(16, 2) labels.shape=torch.Size([16, 1])
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()  # eq里面的两个参数的shape=torch.Size([16])
    acc = correct.sum().item() / len(correct)
    return acc


# 计算模型运行时间
import time
import datetime


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))  # 返回 hh:mm:ss 形式的时间


# 训练模型
def train(model, optimizer):
    t0 = time.time()
    avg_loss, avg_acc = [], []

    model.train()
    for step, batch in enumerate(train_dataloader):

        # 每隔40个batch 输出一下所用时间.
        if step % 10 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
            2].long().to(device)

        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
        loss, logits = output[0], output[1]

        avg_loss.append(loss.item())

        acc = binary_acc(logits, b_labels)
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)  # 大于1的梯度将其设为1.0, 以防梯度爆炸
        optimizer.step()  # 更新模型参数
        scheduler.step()  # 更新learning rate

    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc


# 评估模型
def evaluate(model):
    avg_acc = []
    model.eval()  # 表示进入测试模式

    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
                2].long().to(device)

            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            acc = binary_acc(output[0], b_labels)
            avg_acc.append(acc)
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


# 运行训练模型和评估模型,以及保存模型
for epoch in range(epochs + 1):
    train_loss, train_acc = train(model, optimizer)
    print('epoch={},训练准确率={}，损失={}'.format(epoch, train_acc, train_loss))
    test_acc = evaluate(model)
    print("epoch={},测试准确率={}".format(epoch, test_acc))
    min_loss = 100000  # 随便设置一个比较大的数
    val_loss = train_loss
    if epoch % 3 == 0 and val_loss < min_loss:
        min_loss = val_loss
        print("save model:")
        torch.save(model.state_dict(),
                   f'../model/bert-base-chinese-sighan-all-class-finetuned/pytorch_model.bin')
# 保存最后一个epoch的参数
# torch.save(model.state_dict(), f'../model/bert-base-chinese-sighan-all-class-finetuned/last_pytorch_model.bin')


# if epoch == epochs - 1:
#     torch.save(model.state_dict(), './work/model/bert-base-chinese-sighan14-class-finetuned/pytorch_model.bin')


# 预测
def predict(sen):
    input_id = convert_text_to_token(tokenizer, sen)
    input_token = torch.tensor(input_id).long().to(device)  # torch.Size([128])

    atten_mask = [float(i > 0) for i in input_id]
    attention_token = torch.tensor(atten_mask).long().to(device)  # torch.Size([128])

    output = model(input_token.view(1, -1), token_type_ids=None,
                   attention_mask=attention_token.view(1, -1))  # torch.Size([128])->torch.Size([1, 128])否则会报错
    print(output[0])

    return torch.max(output[0], dim=1)[1]


label = predict('我今天很高行')
print('正确' if label == 1 else '错误')
print('finish the train!')

# label = predict('酒店还可以，接待人员很热情，卫生合格，空间也比较大，不足的地方就是没有窗户')
# print('好评' if label == 1 else '差评')
#
# label = predict('"服务各方面没有不周到的地方, 各方面没有没想到的细节"')
# print('好评' if label == 1 else '差评')
