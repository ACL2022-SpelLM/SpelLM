#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import random
import re

import numpy as np
import torch
# from transformers.modeling_bert import BertForMaskedLM, BertForSequenceClassification
from transformers.tokenization_bert import BertTokenizer

from my_utils import strQ2B

# sys.path.append('../')
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
class_model_path = '../model/bert-base-chinese-sighan-all-class-finetuned'

tokenizer = BertTokenizer.from_pretrained(class_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 读取文件
def readfile(filename):
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
        return content


def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        atten_masks.append(seq_mask)
    return atten_masks


def convert_text_to_token(tokenizer, sentence, limit_size=126):
    tokens = tokenizer.encode(sentence)  # 直接截断
    if len(tokens) < limit_size + 2:  # 补齐（pad的索引号就是0）
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


# 预测
def predict_file_one(file_path, out_path, model):
    # 写结果
    with open(file_path, 'r') as f, open(out_path, 'w') as p:
        count = 0
        lines = f.readlines()
        for i in range(len(lines)):
            # print('sen:', sen)
            sen = lines[i]
            ids = tokenizer.encode('。' + sen)
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
                sen_new.append(predicted_tokens[0])
            p.write(''.join(sen_new) + '\n')
            count += 1
            print(f'{count}:' + ''.join(sen_new))
            # print('*' * 50)
    # 计算指标：


def cal_sen_metric(file_path, out_path, label_path, file_index):
    all_errors = 0  # 所有存在错误的句子
    all_correction = 0  # 所有纠正过的句子
    all_detection = 0  # 所有检测过的字符
    all_correctedly = 0  # 所有正确纠正的句子
    all_detectedly = 0  # 所有正确检测的句子
    all_false_detect = 0

    with open(file_path, 'r') as m, open(out_path, 'r') as p, open(label_path, 'r') as c:
        m_lines, p_lines, c_lines = m.readlines()[0:], p.readlines()[0:], c.readlines()[0:]
        count = 0
        flag_count = 0
        for i in range(len(m_lines)):
            m_line = strQ2B(m_lines[i].strip())
            c_line = strQ2B(c_lines[i].strip())
            p_line = strQ2B(p_lines[i].strip())
            m_line = re.sub(' ', '', m_line)
            # Convenient to keep the sentence length the same
            p_line = re.sub('#', '', p_line)
            p_line = re.sub('\[UNK\]', '.', p_line)
            p_line = re.sub('\[SEP\]', '.', p_line)
            if len(m_line) != len(p_line):
                print('I:', i)
                print(m_line)
                print(p_line)
                print(c_line)
                print('*' * 50)
                continue
            all_errors_list_mistake_id = []  # 记录所有的原始错误
            all_errors_list_correct = []  # 记录原始正确的字符
            all_detection_list = []
            all_correction_list = []
            all_false_detect_list = []
            for j in range(len(m_line)):
                #  bert will often perform letter case conversion
                if len(re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', p_line[j])) != 0 or len(
                        re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', m_line[j])) != 0:
                    continue
                if len(re.findall(r'[?.,!\[\]):\'\~]+', p_line[j])) != 0:
                    continue
                if not (m_line[j] == p_line[j] and m_line[j] == c_line[j]):
                    all_false_detect_list.append(j)

                if m_line[j] != c_line[j]:  # 原始的和正确的不一样则认为是错误的
                    # 记录所有的错误
                    all_errors_list_mistake_id.append(j)
                    all_errors_list_correct.append((j, c_line[j]))
                if j == 0:
                    if p_line[j] == "." or p_line[j] == ",":
                        continue
                if m_line[j] != p_line[j]:  # 原始字符与预测的不同，则说明是检测的字符
                    all_detection_list.append((j))
                    all_correction_list.append((j, p_line[j]))
            if len(all_errors_list_mistake_id) != 0:  # 只要句子中存在错误字符则该句子就是错误的
                all_errors += 1

            if len(all_detection_list) != 0:  #
                all_detection += 1

            if len(all_errors_list_mistake_id) != 0 and sorted(all_errors_list_mistake_id) == sorted(
                    all_detection_list):
                all_detectedly += 1

            if len(all_detection_list) != 0:  # 检测到有错的句子纠正了
                all_correction += 1

            # 检测到有错的句子纠正了，且句子均纠正对了
            if len(all_detection_list) != 0 and sorted(all_errors_list_correct) == sorted(all_correction_list):
                all_correctedly += 1

    check_p = round(all_detectedly / all_detection, 5)
    check_r = round(all_detectedly / all_errors, 5)
    check_F1 = round(2 * (check_p * check_r) / (check_p + check_r), 5)

    s1 = f'sig1{file_index}  metric performance：：：：：：：：：：：：'
    s2 = f'D-P: char_p={all_detectedly}/{all_detection}\t' + str(check_p)
    s3 = f'D-R: char_p={all_detectedly}/{int(all_errors)}\t' + str(check_r)
    s4 = f'D-F1:\t' + str(check_F1)
    m_info = s1 + '\n' + s2 + '\n' + s3 + '\n' + s4 + '\n'
    print(m_info)
    # mr_f.write(s1 + '\n' + s2 + '\n' + s3 + '\n' + s4)

    correct_p = round(all_correctedly / all_correction, 5)
    correct_r = round(all_correctedly / all_errors, 5)
    correct_F1 = round(2 * (correct_p * correct_r) / (correct_p + correct_r), 5)
    s1 = f'C-P  char_p={all_correctedly}/{all_correction}\t' + str(correct_p)
    s2 = f'R-P  char_p={all_correctedly}/{int(all_errors)}\t' + str(correct_r)
    s3 = f'R-F1:\t' + str(correct_F1)
    print(s1 + '\n' + s2 + '\n' + s3)
    m_info = m_info + '\n\n' + s1 + '\n' + s2 + '\n' + s3
    # mr_f.write(s1 + '\n' + s2 + '\n' + s3)
    print('*' * 50)
    return m_info
    # print(f'D-P: char_p={all_detectedly}/{all_detection}\t', check_p)
    # print(f'D-R: char_p={all_detectedly}/{int(all_errors)}\t', check_r)
    # print(f'D-F1:', check_F1)
    #
    # correct_p = round(all_correctedly / all_correction, 5)
    # correct_r = round(all_correctedly / all_errors, 5)
    # correct_F1 = round(2 * (correct_p * correct_r) / (correct_p + correct_r), 5)
    # print(f'C-P: char_p={all_correctedly}/{all_correction}\t', correct_p)
    # print(f'C-R: char_p={all_correctedly}/{int(all_errors)}\t', correct_r)
    # print(f'C-F1:', correct_F1)
    # print('*' * 50)


def cal_character_metric(file_path, out_path, label_path, file_index):
    all_errors = 0  # 所有错误字符
    all_correction = 0  # 所有纠正的字符
    all_detection = 0  # 所有检测的字符
    all_correctedly = 0  # 所有正确纠正的字符
    all_detectedly = 0  # 所有正确检测的字符
    all_false_detect = 0  # 误纠
    # metric_results_character_path = 'metric_results_character.txt'
    with open(file_path, 'r') as m, open(out_path, 'r') as p, open(label_path, 'r') as c:

        m_lines, p_lines, c_lines = m.readlines(), p.readlines(), c.readlines()
        count = 0
        for i in range(len(m_lines)):
            # Keep Chinese and English symbols in the same format
            m_line = strQ2B(m_lines[i].strip())
            c_line = strQ2B(c_lines[i].strip())
            p_line = strQ2B(p_lines[i].strip())
            m_line = re.sub(' ', '', m_line)
            p_line = re.sub('#', '', p_line)
            # Convenient to keep the sentence length the same
            p_line = re.sub('\[UNK\]', '.', p_line)
            p_line = re.sub('\[SEP\]', '.', p_line)
            if len(m_line) != len(p_line):
                print('I:', i)
                print(m_line)
                print(p_line)
                print(c_line)
                print('*' * 50)
                continue
            count += 1
            for j in range(len(m_line)):
                # Because bert will often perform letter case conversion
                if len(re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', p_line[j])) != 0 or len(
                        re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', m_line[j])) != 0:
                    continue
                if len(re.findall(r'[?.,!\[\]):\'\~]+', p_line[j])) != 0:
                    continue
                if m_line[j] != c_line[j]:  # 原始的和正确的不一样则认为是错误的
                    all_errors += 1
                if j == 0:
                    if p_line[j] == "." or p_line[j] == ",":
                        continue
                if m_line[j] == p_line[j] and m_line[j] == c_line[j]:  # 这地方没错，属于误纠
                    all_false_detect += 1

                if m_line[j] != p_line[j]:  # 原始字符与预测的不同，则说明是检测的字符
                    all_detection += 1
                    if m_line[j] != c_line[j]:  # 本来有错，且检测到有错
                        all_detectedly += 1

                if m_line[j] != c_line[j]:
                    if m_line[j] != p_line[j]:  # 本来有错的地方,纠正了
                        all_correction += 1
                    if m_line[j] != p_line[j] and p_line[j] == c_line[j]:  # 本来有错的地方，纠正且纠正对了
                        all_correctedly += 1

    check_p = round(all_detectedly / all_detection, 5)
    check_r = round(all_detectedly / all_errors, 5)
    check_F1 = round(2 * (check_p * check_r) / (check_p + check_r), 5)

    s1 = f'sig1{file_index}  metric performance：：：：：：：：：：：：'
    s2 = f'D-P: char_p={all_detectedly}/{all_detection}\t' + str(check_p)
    s3 = f'D-R: char_p={all_detectedly}/{int(all_errors)}\t' + str(check_r)
    s4 = f'D-F1:\t' + str(check_F1)
    m_info = s1 + '\n' + s2 + '\n' + s3 + '\n' + s4 + '\n'
    print(m_info)
    # mr_f.write(s1 + '\n' + s2 + '\n' + s3 + '\n' + s4)

    correct_p = round(all_correctedly / all_correction, 5)
    correct_r = round(all_correctedly / all_errors, 5)
    correct_F1 = round(2 * (correct_p * correct_r) / (correct_p + correct_r), 5)
    s1 = f'C-P  char_p={all_correctedly}/{all_correction}\t' + str(correct_p)
    s2 = f'R-P  char_p={all_correctedly}/{int(all_errors)}\t' + str(correct_r)
    s3 = f'R-F1:\t' + str(correct_F1)
    print(s1 + '\n' + s2 + '\n' + s3)
    m_info = m_info + '\n' + s1 + '\n' + s2 + '\n' + s3 + '\n\n'
    # mr_f.write(s1 + '\n' + s2 + '\n' + s3)
    print('*' * 50)
    return m_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='self-attention score for word in sentence ')
    parser.add_argument('--with_error', '-with_error', type=bool, default=True,
                        help='with_error is character level metric ,else is sentence level')
    # parser.add_argument('--test_data', '-t', type=str, default=None,
    #                     help='')
    # parser.add_argument('--index', '-i', type=int, default=False,
    #                     help='the word index in sentence ')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    with_error = args.with_error
    # with_error = True
    print('start to predict!')
    f_path = '../data/test_data/'
    m_path = '../model/bert-base-chinese-q-layer/'
    m_infos = []
    for i in range(3, 6):
        if i == 3:
            model_path = m_path + "best_pytorch_model_6.bin"                 
        else:
            model_path = m_path + "best_pytorch_model_13.bin"
        model = torch.load(model_path)

        if with_error:  # character level
            file_path = f_path + f'sighan1{i}_test/sig_1{i}_mistake_test_with_error.txt'
            label_path = f_path + f'sighan1{i}_test/sig_1{i}_correct_test_with_error.txt'
            out_path = f'../test_out/sighan1{i}_test_result/sig1{i}_ig_test_with_error.txt'
            predict_file_one(file_path, out_path, model)
            m_info = cal_character_metric(file_path, out_path, label_path, i)
            m_infos.append(m_info)
        else:  # sentence level
            file_path = f_path + f'sighan1{i}_test/sig_1{i}_mistake_test.txt'
            label_path = f_path + f'sighan1{i}_test/sig_1{i}_correct_test.txt'
            out_path = f'../test_out/sighan1{i}_test_result/sig1{i}_ig_test.txt'
            predict_file_one(file_path, out_path, model)
            m_info = cal_sen_metric(file_path, out_path, label_path, i)
            m_infos.append(m_info)
    if with_error:  # character level
        metric_results_path = 'metric_results_character.txt'
    else:  # sentence level
        metric_results_path = 'metric_results_sentence.txt'
    # if not os.path.exists(metric_results_path):
    #     os.makedirs(metric_results_path)
    with open(metric_results_path, 'w') as m_info_f:
        m_info_f.write('\n'.join(m_infos))
