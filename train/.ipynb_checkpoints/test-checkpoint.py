import random

import numpy as np
import torch
import sys
# from transformers.modeling_bert import BertForMaskedLM, BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer
# sys.path.append('../')



random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# 读取文件
def readfile(filename):
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
        return content


# attention_masks,在一个文本中，如果是pad符号则是0，否则就是1
# 建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks


class_model_path = '../model/bert-base-chinese-sighan-all-class-finetuned'

tokenizer = BertTokenizer.from_pretrained(class_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 预测
def predict(sens):
    for sen in sens:
        print('sen:', sen)
        ids = tokenizer.encode(sen)
        b_input_ids = torch.tensor([ids]).to(device)
        logits = model(input_token=b_input_ids,
                             attention_mask=None)
        prediction_scores = logits
        pre = torch.softmax(prediction_scores[0], -1)
        top_info = torch.topk(pre, k=5)
        scores = top_info[0]
        ids = top_info[1]
        sen_new = []
        for i in range(1, ids.size()[0] - 1):
            predicted_tokens = tokenizer.convert_ids_to_tokens(ids[i])
            # print(predicted_tokens)
            # 取top1的结果，并写到文件中
            # 取top1的结果，并写到文件中
            sen_new.append(predicted_tokens[0])
        print('pre: ' + ''.join(sen_new) + '\n')
    print('a')

def convert_text_to_token(tokenizer, sentence, limit_size=126):
    tokens = tokenizer.encode(sentence)  # 直接截断
    if len(tokens) < limit_size + 2:  # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens

# 预测
def predict_file(file_path, out_path, correct_path):
    with open(file_path, 'r') as f, open(out_path, 'w') as p, open(correct_path, 'r') as c:
        count = 0
        lines = f.readlines()
        c_lines = c.readlines()
        input_ids = [convert_text_to_token(tokenizer, sen) for sen in lines]
        print(input_ids)
        input_tokens = torch.tensor(input_ids)
        test_data = TensorDataset(input_tokens)
        test_sampler = RandomSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

        for step, batch in enumerate(test_dataloader):
            b_input_ids = batch[0].long().to(device)
            output = model(b_input_ids, attention_mask=None)
            # print('sen:', sen)
#             print(output)
#             print(outpu.shape)
            prediction_scores = logits
            pre = torch.softmax(prediction_scores[0], -1)
            print(pre.shape)
            top_info = torch.topk(pre, k=5)
            scores = top_info[0]
            ids = top_info[1].squeeze(dim=0)
#             print(ids)
#             print(ids.size())
#             print(ids.size()[0])
            
            sen_new = []
            for i in range(1, ids.size()[0] - 1):
                predicted_tokens = tokenizer.convert_ids_to_tokens(ids[i])
                print(predicted_tokens)
                # 取top1的结果，并写到文件中
                # 取top1的结果，并写到文件中
                sen_new.append(predicted_tokens[0])

            p.write(''.join(sen_new) + '\n')
            count += 1
            print(f'{count}:' + ''.join(sen_new) + '\n')
            print('*' * 50)
        print('a')

# 预测
def predict_file_one(file_path, out_path, correct_path,model):
    with open(file_path, 'r') as f, open(out_path, 'w') as p, open(correct_path, 'r') as c:
        count = 0
        lines = f.readlines()
        c_lines = c.readlines()
        for i in range(len(lines)):
            # print('sen:', sen)
            sen = lines[i]
            # c_line = c_lines[i]

            # print('原始：', sen)
            # print('正确：', c_line)

            ids = tokenizer.encode('。'+sen)
            b_input_ids = torch.tensor([ids]).to(device)
            logits = model(b_input_ids,
                                 attention_mask=None)
            prediction_scores = logits[0]
            pre = torch.softmax(prediction_scores[0], -1)
            top_info = torch.topk(pre, k=5)
            scores = top_info[0]
            ids = top_info[1].squeeze()
            sen_new = []
            for i in range(2, ids.size()[0] - 1):
                predicted_tokens = tokenizer.convert_ids_to_tokens(ids[i])
                # print(predicted_tokens)
                # 取top1的结果，并写到文件中
                # 取top1的结果，并写到文件中
                sen_new.append(predicted_tokens[0])

            p.write(''.join(sen_new) + '\n')
            count += 1
            print(f'{count}:' + ''.join(sen_new) + '\n')
            # print('*' * 50)
        print('a')

if __name__ == '__main__':
    print('进入预测阶段：')
    # 听起来是一份很好的公司。又意思又很多钱。
    # 因为我下个礼拜有很重要的考试，所以得要好好地复习。
    # 敬祝身体建慷。
    # 我很高兴听说你已经找到了工作。我真恭喜妳阿。
    # 听说你准备开一个祝会，那天我不能跟你参加。我很抱见阿。
    # 我知道，你很久以前找工作很幸苦。
    # 我很想参家你的舞会因为你的舞会总是很好玩。
    # 我下个星期非常忙所以我不能参家你的舞会。
    # 我下个星期天我妈妈回来台湾。
    # 因为我不能参家你的舞会所以我要请你跟我去吃晚饭再看电影。

    sens = ['坐路差不多十分钟，我们到了。', '今天天七很好。', '真麻烦你了。希望你们好好的跳无。',
            '下个星期，我跟我朋唷打算去法国玩儿。', '在公车上有很多人，所以我们没有位子可以座。',
            '我以前想要高诉你，可是我忘了。', '吃了早菜以后他去上课。', '我今天很高行。', '我今天很开新。']
    sens_train = [' 听起来是一份很好的公司。又意思又很多钱。', '因为我下个礼拜有很重要的考试，所以得要好好地复习。',
                  ' 敬祝身体建慷。', '我很高兴听说你已经找到了工作。我真恭喜妳阿。', '听说你准备开一个祝会，那天我不能跟你参加。我很抱见阿。',
                  '我知道，你很久以前找工作很幸苦。', '我很想参家你的舞会因为你的舞会总是很好玩。', '我下个星期非常忙所以我不能参家你的舞会。',
                  '我下个星期天我妈妈回来台湾。', ' 因为我不能参家你的舞会所以我要请你跟我去吃晚饭再看电影。', '我今天很高兴']
#     predict(sens)
    file_path = '../data/sig_13_mistake_test_gcn.txt'
   
    correct_path = '../data/sig_13_correct_test_gcn.txt'
    
  
    for i in range(2,29):
        model_path=f"../save/bert_ig_300k_q_layer/best_pytorch_model_{i}.bin"
        out_path = f'../test_out/13_ig_300k_q_layer/sig13_ig_test_{i}.txt'
        model =torch.load(model_path)
        predict_file_one(file_path, out_path, correct_path,model)
    

#     file_path = '../data/sig_14_mistake_test_gcn.txt'
#     out_path = './14_test/train_ig_29.txt'
#     correct_path = '../data/sig_14_correct_test_gcn.txt'
#     predict_file_one(file_path, out_path, correct_path)

# label = predict('我今天很高行')
# print('好评' if label == 1 else '差评')

# label = predict('酒店还可以，接待人员很热情，卫生合格，空间也比较大，不足的地方就是没有窗户')
# print('好评' if label == 1 else '差评')
#
# label = predict('"服务各方面没有不周到的地方, 各方面没有没想到的细节"')
# print('好评' if label == 1 else '差评')
# /practice/self-attention/integrated_gradients/model
