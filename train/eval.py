# -*- coding: GBK -*-
import re


def strQ2B(ustring):
    """中文特殊符号转英文特殊符号"""
    # 中文特殊符号批量识别

    pattern = re.compile('[，。：“”【】《》？；、（）‘’『』「」﹃﹄〔〕—·]')

    # re.compile: 编译一个正则表达式模式，返回一个模式（匹配模式）对象。
    # [...]用于定义待转换的中文特殊符号字符集

    fps = re.findall(pattern, ustring)

    # re.findall: 搜索string，以列表形式返回全部能匹配的子串。

    # 对有中文特殊符号的文本进行符号替换

    if len(fps) > 0:
        ustring = ustring.replace('，', ',')
        ustring = ustring.replace('。', '.')
        ustring = ustring.replace('：', ':')
        ustring = ustring.replace('“', '"')
        ustring = ustring.replace('”', '"')
        ustring = ustring.replace('【', '[')
        ustring = ustring.replace('】', ']')
        ustring = ustring.replace('《', '<')
        ustring = ustring.replace('》', '>')
        ustring = ustring.replace('？', '?')
        ustring = ustring.replace('；', ':')
        ustring = ustring.replace('、', ',')
        ustring = ustring.replace('（', '(')
        ustring = ustring.replace('）', ')')
        ustring = ustring.replace('‘', "'")
        ustring = ustring.replace('’', "'")
        ustring = ustring.replace('’', "'")
        ustring = ustring.replace('『', "[")
        ustring = ustring.replace('』', "]")
        ustring = ustring.replace('「', "[")
        ustring = ustring.replace('」', "]")
        ustring = ustring.replace('﹃', "[")
        ustring = ustring.replace('﹄', "]")
        ustring = ustring.replace('〔', "{")
        ustring = ustring.replace('〕', "}")
        ustring = ustring.replace('—', "-")
        ustring = ustring.replace('·', ".")

    """全角转半角"""
    # 转换说明：
    # 全角字符unicode编码从65281~65374 （十六进制 0xFF01 ~ 0xFF5E）
    # 半角字符unicode编码从33~126 （十六进制 0x21~ 0x7E）
    # 空格比较特殊，全角为 12288（0x3000），半角为 32（0x20）
    # 除空格外，全角/半角按unicode编码排序在顺序上是对应的（半角 + 0x7e= 全角）,所以可以直接通过用+-法来处理非空格数据，对空格单独处理。

    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def cal_pre(n):
    detection_chars = []
    # correctedly_chars=[]
    # detectedly_chars=[]

    all_errors = 0  # 所有错误字符
    all_correction = 0  # 所有纠正的字符
    all_detection = 0  # 所有检测的字符
    all_correctedly = 0  # 所有正确纠正的字符
    all_detectedly = 0  # 所有正确检测的字符

    # mask_new_{file}.txt
    with open('../data/sig_15_mistake_test_gcn.txt', 'r') as m, open('trian_ig_test_pre.txt', 'r') as p, open(
            '../data/sig_15_correct_test_gcn.txt', 'r') as c:
        m_lines, p_lines, c_lines = m.readlines()[0:n], p.readlines(), c.readlines()[0:n]
        count = 0
        for i in range(len(m_lines)):

            #         m_lines[i] = re.sub('||', '', m_lines[i])
            m_line = strQ2B(m_lines[i].strip())
            c_line = strQ2B(c_lines[i].strip())
            p_line = strQ2B(p_lines[i].strip())
            m_line = re.sub(' ', '', m_line)
            p_line = re.sub('#', '', p_line)
            p_line = re.sub('\[UNK\]', '.', p_line)

            p_line = re.sub('\[SEP\]', '.', p_line)
            #         p_line = re.sub('||', '', p_line)
            p_line = re.sub(' ', '', p_line)
            if len(m_line) != len(p_line):
                print('I:', i)
                print(m_line)
                print(p_line)
                print(c_line)
                print('*' * 50)
                continue
            #         print(m_line)
            #         print(c_line)
            #         print(p_line)
            count += 1
            for j in range(len(m_line)):
                if len(re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', p_line[j])) != 0 or len(
                        re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', m_line[j])) != 0:
                    continue
                if len(re.findall(r'[?.,!]+', p_line[j])) != 0:
                    continue
                if j == 0:
                    if p_line[j] == "." or p_line[j] == ",":
                        continue
                if m_line[j] != c_line[j]:  # 原始的和正确的不一样则认为是错误的
                    all_errors += 1
                if m_line[j] != p_line[j]:  # 原始字符与预测的不同，则说明是检测的字符
                    all_detection += 1
                    if m_line[j] != c_line[j]:  # 本来有错，且检测到有错
                        all_detectedly += 1
                if m_line[j] != c_line[j]:
                    if m_line[j] != p_line[j]:  # 本来有错的地方,纠正了
                        all_correction += 1
                    if m_line[j] != p_line[j] and p_line[j] == c_line[j]:  # 本来有错的地方，纠正且纠正对了
                        all_correctedly += 1

    # 得到所有的错误字符个数：
    # all_errors=491
    # print('检测的:',''.join(detection_chars))
    # print('正确检测的:',''.join(detectedly_chars))
    # print('正确纠正的:',''.join(correctedly_chars))
    print('count', count)
    print('all_errors', all_errors)
    print('all_detection', all_detection)
    print('all_detectedly', all_detectedly)
    print('all_correction', all_correction)
    print('all_correctedly', all_correctedly)

    check_p = round(all_detectedly / all_detection, 5)
    check_r = round(all_detectedly / all_errors, 5)
    check_F1 = round(2 * (check_p * check_r) / (check_p + check_r), 5)
    print(f'检测精确率: 正确检测出/所有检测出的字符 char_p={all_detectedly}/{all_detection}\t', check_p)
    print(f'检测召回率: 正确检测出/所有错误的字符   char_p={all_detectedly}/{int(all_errors)}\t', check_r)
    print(f'检测的F1值:', check_F1)

    correct_p = round(all_correctedly / all_correction, 5)
    correct_r = round(all_correctedly / all_errors, 5)
    correct_F1 = round(2 * (correct_p * correct_r) / (correct_p + correct_r), 5)
    print(f'纠正精确率: 正确纠正的/所有纠正的字符  char_p={all_correctedly}/{all_correction}\t', correct_p)
    print(f'纠正召回率: 正确纠正的/所有错误的字符  char_p={all_correctedly}/{int(all_errors)}\t', correct_r)
    print(f'纠正的F1值:', correct_F1)
    print('*' * 50)
