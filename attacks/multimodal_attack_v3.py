import json
import os
import re
import pickle
import jieba
import numpy as np
import torch
import argparse
from PIL import Image
from matplotlib import pyplot as plt
from gensim.models import FastText
from snownlp import SnowNLP
from torchvision.transforms import transforms
from transformers import CLIPProcessor, CLIPModel

from models.EANN.process_data_weibo import clean_str_sst
from scripts.EANN_attack import process_text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#fasttext_model = FastText.load_fasttext_format('../utils/cc.zh.300.bin')

file_path = "../data/weibo/tweets/updated_test_tweet_data_clean.json"

# 词性排序的优先级
POS_ORDER = {
    'JJ': 1,  # 形容词 (Adjective)
    'VA': 1,  # 形容词/动词化 (Adjective/Verb)
    'VB': 2,  # 动词 (Verb)
    'VV': 2,  # 动词 (Verb)
    'VBD': 2,  # 动词过去式 (Verb, past tense)
    'VBN': 2,  # 动词过去分词 (Verb, past participle)
    'VBG': 2,  # 动词的现在分词 (Verb, gerund)
    'VE': 2,  # 动词（表示状态的）(Verb, state)
    'VBZ': 2, # 动词第三人称单数 (Verb, third person singular)
    'VC': 3,  # 动词复合形式 (Verb Compound)
    'NN': 4,  # 名词 (Noun)
    'NNS': 4,  # 名词复数 (Noun plural)
    'OD': 5,  # 其他名词 (Other noun)
    'FW': 6,  # 外来语 (Foreign word)
    'DER': 7, # 派生词 (Derived word)
    'NR': 8,  # 专有名词 (Proper noun)
    'NNP': 8,  # 专有名词 (Proper noun)
    'NT': 9,  # 名词类别 (Noun type)
    'LC': 10,  # 类别词 (Category word)
    'RB': 11,  # 副词 (Adverb)
    'AD': 12,  # 副词（另一类）(Adverb variant)
    'AS': 13,  # 比较副词 (Adverb, comparative)
    'MD': 14,  # 情态动词 (Modal verb)
    'MSP': 15, # 主语短语 (Subject phrase)
    'PN': 16,  # 名词代词 (Pronoun noun)
    'M': 17,  # 量词 (Measure word)
    'LS': 18,  # 数词 (Numeral)
    'CD': 18,  # 数词 (Cardinal number)
    'DT': 19,  # 限定词 (Determiner)
    'DEG': 20,  # 结构助词 (Structural particle)
    'NOI': 21,  # 名词短语指示词 (Noun phrase indicator)
    'ETC': 22,  # 等等 (Etc.)
    'URL': 23,  # 网络地址 (URL)
    'SB': 24,  # 主语名词短语 (Subject noun phrase)
    'BA': 25,  # “把”字句 (Ba Construction)
    'DEC': 26,  # 定语从句标记 (Attributive clause marker)
    'DEV': 27,  # 发展助词 (Developmental particle)
    'LB': 28,  # 方位词或标记性词语 (Locative word)
    'ON': 29,  # 方位词 (Position word)
    'SP': 30,  # 语气词 (Modal particle)
    'IC': 31,  # 感叹词 (Interjection)
    'IJ': 32,  # 感叹词 (Interjection)
    'CS': 33,  # 连词 (Conjunction)
    'C': 34,  # 连词 (Conjunction)
    'P': 35,  # 介词 (Preposition)
    'IN': 36,  # 介词或连词 (Preposition or Conjunction)
    'CC': 37,  # 并列连词 (Coordinating conjunction)
    'PRP': 38,  # 代词 (Pronoun)
    'PU': 39,  # 标点符号 (Punctuation)
}
stop_word_pos = [
    'LS',  # 数词 (Numeral)
    'CD',  # 数词 (Cardinal number)
    'DT',  # 限定词 (Determiner)
    'DEG',  # 结构助词 (Structural particle)
    'NOI',  # 名词短语指示词 (Noun phrase indicator)
    'ETC',  # 等等 (Etc.)
    'URL',  # 网络地址 (URL)
    'SB',  # 主语名词短语 (Subject noun phrase)
    'BA',  # “把”字句 (Ba Construction)
    'DEC',  # 定语从句标记 (Attributive clause marker)
    'DEV',  # 发展助词 (Developmental particle)
    'LB',  # 方位词或标记性词语 (Locative word)
    'ON',  # 方位词 (Position word)
    'SP',  # 语气词 (Modal particle)
    'IC',  # 感叹词 (Interjection)
    'IJ',  # 感叹词 (Interjection)
    'CS',  # 连词 (Conjunction)
    'C',  # 连词 (Conjunction)
    'P',  # 介词 (Preposition)
    'IN',  # 介词或连词 (Preposition or Conjunction)
    'CC',  # 并列连词 (Coordinating conjunction)
    'PRP',  # 代词 (Pronoun)
    'PU'  # 标点符号 (Punctuation)
]


def extract_tweet_data(tweet_id, attack_step, label):
    try:
        # 打开并读取 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            tweets_data = json.load(f)

        # 在文件中查找对应的 tweet_id
        tweet_data = next((t for t in tweets_data if t.get("tweet_id") == tweet_id), None)

        if not tweet_data:
            return {
                "sentence_id": None,
                "sentence_content": None,
                "image_id": None,
                "image_content": None,
            }

        # 获取sorted_similarities字段
        sorted_similarities = tweet_data.get("sorted_similarities", [])
        if not sorted_similarities:
            return {
                "sentence_id": None,
                "sentence_content": None,
                "image_id": None,
                "image_content": None,
            }

        # 根据标签选择需要的相似度数据
        if label == 0:  # 真新闻：取第一条
            selected_similarity = sorted_similarities[0]
            #selected_similarity = sorted_similarities[attack_step % len(sorted_similarities)]
        else:  # 假新闻：根据攻击步数逐步选择
            selected_similarity = sorted_similarities[attack_step % len(sorted_similarities)]

        # 获取相似度数据对应的 sentence_id 和 image_id
        sentence_id = selected_similarity.get("sentence_id")
        image_id = selected_similarity.get("image_id")

        sentence_content = tweet_data.get("sentences", {}).get(sentence_id, {}).get("text")
        image_content = tweet_data.get("images", {}).get(image_id)

        return {
            "sentence_id": sentence_id,
            "sentence_content": sentence_content,
            "image_id": image_id,
            "image_content": image_content
        }

    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return {
            "sentence_id": None,
            "sentence_content": None,
            "image_id": None,
            "image_content": None,
            "sorted_similarities": [],
        }

# 定义一个函数来判断词是否为特殊字符或无意义的词
def is_meaningless_word(word):
    """
    判断词是否是特殊字符、数字或无意义的词。
    """
    # 检查词是否只包含特殊字符或数字
    if re.match(r'^[^\w\u4e00-\u9fa5]+$', word):  # 只包含非字母、非汉字字符
        return True
    '''# 检查词是否是无意义词
    if word in meaningless_words:
        return True'''
    # 检查词是否是仅由空格组成
    if word.strip() == "":
        return True
    if word in ('O', '&nbsp', '__', '”', '"', '“', 'Đứ', 'nbsp', 'ǐ', 'ps'):
        return True
    # 如果词符合任何条件，则认为是无意义的词
    return False

def get_word_and_pos(tweet_id, sentence_id, label, attack_history):
    try:
        # 打开并读取 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            tweets_data = json.load(f)

        # 查找包含给定 tweet_id 的推文
        tweet_data = next((t for t in tweets_data if t["tweet_id"] == tweet_id), None)

        if not tweet_data:
            print(f"未找到 tweet_id: {tweet_id}")
            return {
                "word": None,
                "start_pos": None,
            }

        # 查找包含给定 sentence_id 的分句
        sentence_data = tweet_data.get("sentences", {}).get(sentence_id, None)

        if not sentence_data:
            print(f"未找到 sentence_id: {sentence_id} 在 tweet_id: {tweet_id} 中")
            return {
                "word": None,
                "start_pos": None,
            }

        # 获取分句的 tokenized_data
        tokenized_data = sentence_data.get("tokenized_data", [])

        if not tokenized_data:
            return {
                "word": None,
                "start_pos": None,
            }

        # 真新闻处理：从前往后找符合三个条件的词
        if label == 0:
            for token in tokenized_data:
                word = token.get("word")
                start_pos = token.get("start_pos")
                pos_tag = token.get("pos_tag")
                # 判断词性、是否是无意义词、是否已经在攻击历史中
                if (pos_tag not in stop_word_pos and not is_meaningless_word(word) and (sentence_id, start_pos) not in attack_history):
                    attack_history.append((sentence_id, start_pos))  # 记录攻击历史
                    return {
                        "word": word,
                        "start_pos": start_pos,
                    }
                elif pos_tag not in stop_word_pos and not is_meaningless_word(word):
                    return {
                        "word": word,
                        "start_pos": start_pos,
                    }
                elif not is_meaningless_word(word):
                    return {
                        "word": word,
                        "start_pos": start_pos,
                    }
        # 假新闻处理：从后往前找符合三个条件的词
        else:
            for token in reversed(tokenized_data):
                word = token.get("word")
                start_pos = token.get("start_pos")
                pos_tag = token.get("pos_tag")
                # 判断词性、是否是无意义词、是否已经在攻击历史中
                if (pos_tag not in stop_word_pos and not is_meaningless_word(word) and (sentence_id, start_pos) not in attack_history):
                    attack_history.append((sentence_id, start_pos))  # 记录攻击历史
                    return {
                        "word": word,
                        "start_pos": start_pos,
                    }
                elif pos_tag not in stop_word_pos and not is_meaningless_word(word):
                    return {
                        "word": word,
                        "start_pos": start_pos,
                    }
                elif not is_meaningless_word(word):
                    return {
                        "word": word,
                        "start_pos": start_pos,
                    }

        # 如果没有找到符合条件的词，返回最后一个词
        word = tokenized_data[-1].get("word")
        start_pos = tokenized_data[-1].get("start_pos")
        return {
            "word": word,
            "start_pos": start_pos,
        }

    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return {
            "word": None,
            "start_pos": None,
        }
    except json.JSONDecodeError:
        print(f"文件内容不是有效的 JSON 格式: {file_path}")
        return {
            "word": None,
            "start_pos": None,
        }

def get_sentiment_score(word):
    """
    使用 SnowNLP 来计算单词的情感分数，返回一个浮动在 [0, 1] 之间的分数。
    0 表示负面情感，1 表示正面情感。
    """
    # 使用 SnowNLP 进行情感分析
    print(f"情感word: {word}")
    s = SnowNLP(word)
    return s.sentiments  # 返回情感得分，范围 [0, 1]

def get_combined_synonyms(word, model, k):
    # 使用jieba对输入的中文单词进行分词
    print(f"同义word: {word}")
    word = clean_str_sst(word)
    words = list(jieba.cut(word))

    # 用于存储每个分词的同义词
    synonym_lists = []

    # 获取每个分词的前k个同义词及其相似度
    for sub_word in words:
        synonyms = model.wv.most_similar(sub_word, topn=k)
        synonym_lists.append(synonyms)

    # 初始化最终的k个同义词及其相似度
    combined_synonyms = []
    combined_similarities = []

    # 将每个分词的第i个同义词进行拼接
    for i in range(k):
        combined_word = ""
        total_similarity = 0.0
        for j in range(len(words)):
            synonym, similarity = synonym_lists[j][i]
            combined_word += synonym  # 拼接同义词
            total_similarity += similarity  # 累加相似度
        combined_synonyms.append(combined_word)
        combined_similarities.append(total_similarity)

    # 返回拼接后的同义词和对应的相似度
    return combined_synonyms, combined_similarities

def find_adv_word(word, model, k, label):
    """
    利用 FastText 模型找出输入词的 top-k 个同义词，并对其进行情感分析，返回最低或最高情感分的同义词。

    参数:
    word: 输入的词语。
    model: 已加载的 FastText 模型。
    k: 返回同义词的个数。
    label: 0 表示真新闻，1 表示假新闻。

    返回:
    根据标签返回情感分最低或最高的同义词。
    """
    # 获取与输入词最相似的 top-k 个词
    combined_synonyms, combined_similarities = get_combined_synonyms(word, model, k)
    neighbors = list(zip(combined_synonyms, combined_similarities))

    # 过滤出有意义的同义词并计算情感分数
    word_sentiments = []
    for similar_word, similarity in neighbors:
        if not is_meaningless_word(similar_word):  # 检查是否为无意义词
            sentiment_score = get_sentiment_score(similar_word)
            word_sentiments.append((similar_word, sentiment_score))

    # 如果所有同义词都是无意义词，返回 None
    if not word_sentiments:
        return None

    # 根据标签选择情感分最低或最高的词
    if label == 0:
        # 真新闻：返回情感分最低的词
        selected_word = min(word_sentiments, key=lambda x: x[1])
    else:
        # 假新闻：返回情感分最高的词
        selected_word = max(word_sentiments, key=lambda x: x[1])

    return selected_word

def extract_words_and_positions(tweet_id, sentence_id):
    try:
        # 打开并读取 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            tweet_data = json.load(f)

        # 查找包含给定 tweet_id 的推文
        tweet = next((t for t in tweet_data if t.get("tweet_id") == tweet_id), None)

        if not tweet:
            print(f"未找到 tweet_id: {tweet_id}")
            return None

        # 查找包含给定 sentence_id 的分句
        sentence = tweet.get("sentences", {}).get(sentence_id, None)

        if not sentence:
            print(f"未找到 sentence_id: {sentence_id} 在 tweet_id: {tweet_id} 中")
            return None

        # 获取分句的 tokenized_data
        tokenized_data = sentence.get("tokenized_data", [])

        # 提取所有 word 和 start_pos
        words_and_positions = [(token['word'], token['start_pos']) for token in tokenized_data]

        return words_and_positions

    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"文件内容不是有效的 JSON 格式: {file_path}")
        return None

def replace_word_in_sentence(words_and_positions, adv_word, start_pos):
    """
    根据start_pos替换words_and_positions中的词。

    参数:
    words_and_positions: extract_words_and_positions函数的输出，包含(word, start_pos)元组的列表。
    adv_word: 要替换的词。
    start_pos: 要替换的词的起始位置。

    返回:
    修改后的words_and_positions列表。
    """
    # 找到要替换的词的索引
    for i, (word, pos) in enumerate(words_and_positions):
        if pos == start_pos:
            # 替换词
            words_and_positions[i] = (adv_word, pos)
            break
    else:
        # 如果没有找到对应的start_pos，打印错误信息
        print(f"未找到起始位置: {start_pos} 以替换词")

    return words_and_positions

def replace_word_in_json(tweet_id, sentence_id, adv_word, start_pos, adv_sentiments):
    """
    在 JSON 文件中根据指定的 tweet_id 和 sentence_id 替换指定位置的词。

    参数:
    file_path: JSON 文件的路径。
    tweet_id: 要替换词语的 tweet_id。
    sentence_id: 要替换词语的 sentence_id。
    adv_word: 要替换的词。
    start_pos: 要替换的词的起始位置。
    adv_sentiments: 替换词的情感得分。
    pos_tag: 替换词的词性（如 'JJ', 'VV', 等）。

    返回:
    返回修改后的 JSON 数据。
    """
    try:
        # 打开并读取 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            tweet_data = json.load(f)

        # 查找包含给定 tweet_id 的推文
        tweet = next((t for t in tweet_data if t.get("tweet_id") == tweet_id), None)

        if not tweet:
            print(f"未找到 tweet_id: {tweet_id}")
            return None

        # 查找包含给定 sentence_id 的分句
        sentence = tweet.get("sentences", {}).get(str(sentence_id), None)

        if not sentence:
            print(f"未找到 sentence_id: {sentence_id} 在 tweet_id: {tweet_id} 中")
            return None

        # 获取分句的 tokenized_data
        tokenized_data = sentence.get("tokenized_data", [])

        # 将修改后的词和情感得分重新写回 tokenized_data
        word_found = False
        for token in tokenized_data:
            if token['start_pos'] == start_pos:
                token['word'] = adv_word
                token['sentiments'] = adv_sentiments  # 更新情感得分
                word_found = True
                break  # 假设每次只替换一个词

        if not word_found:
            print(f"未找到起始位置为 {start_pos} 的词")
            return None

        # 按照 sentiments 从高到低排序，情感得分相同的情况下按照词性排序
        tokenized_data = sorted(
            tokenized_data,
            key=lambda x: (x['sentiments'], POS_ORDER.get(x.get('pos_tag', ''), float('inf'))),
            reverse=True
        )

        # 将更新后的 tokenized_data 重新写回 JSON 数据
        sentence['tokenized_data'] = tokenized_data

        # 将更新后的数据写回 JSON 文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tweet_data, f, ensure_ascii=False, indent=4)

        return tweet_data

    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"文件内容不是有效的 JSON 格式: {file_path}")
        return None
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        return None

def sort_and_construct_sentence(modified_result):
    """
    根据单词的起始位置从小到大排序，并构建一个句子。

    参数:
    modified_result: replace_word_in_sentence函数的输出，包含(word, start_pos)元组的列表。

    返回:
    根据单词起始位置排序后构建的句子。
    """
    # 根据 start_pos 排序
    sorted_result = sorted(modified_result, key=lambda x: x[1])

    # 构建句子
    sentence = ''.join(word for word, _ in sorted_result)

    return sentence

def extract_sentences_text(tweet_id):
    try:
        # 打开并读取 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            tweets_data = json.load(f)

        # 在文件中查找对应的 tweet_id
        tweet_data = next((t for t in tweets_data if t.get("tweet_id") == tweet_id), None)

        if not tweet_data:
            print(f"未找到 tweet_id: {tweet_id}")
            return []

        # 提取 sentences 中的所有 sentence_id 和 text
        sentences_text = []
        sentences = tweet_data.get("sentences", {})
        for sentence_id, sentence_info in sentences.items():
            sentence_text = sentence_info.get("text", "")
            start_pos = sentence_info.get("start_pos", "")
            sentences_text.append({
                "sentence_id": sentence_id,
                "text": sentence_text,
                "start_pos": start_pos
            })

        return sentences_text

    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"文件内容不是有效的 JSON 格式: {file_path}")
        return []

def replace_sentence_text(sentences_text, sorted_sentence, sentence_id):
    """
    用sorted_sentence替换sentences_text里id为sentence_id的文本。

    参数:
    sentences_text: 包含所有句子的ID和文本的列表。
    sorted_sentence: 要替换的新句子文本。
    sentence_id: 要被替换的句子ID。

    返回:
    更新后的sentences_text列表。
    """
    # 遍历sentences_text列表，找到匹配的sentence_id并替换文本
    for sent in sentences_text:
        if sent['sentence_id'] == sentence_id:
            sent['text'] = sorted_sentence
            break  # 找到匹配的ID后不再继续遍历

    return sentences_text

def replace_sentence_in_json_file(tweet_id, sentence_id, sorted_sentence):
    """
    在 JSON 文件中替换指定 tweet_id 和 sentence_id 的句子文本。

    参数:
    file_path: JSON 文件的路径。
    tweet_id: 要修改的 tweet_id。
    sentence_id: 要替换的 sentence_id。
    sorted_sentence: 新的句子文本。

    返回:
    成功时返回 True，失败时返回 False。
    """
    try:
        # 打开并读取 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            tweets_data = json.load(f)

        # 查找 tweet_id 对应的数据
        tweet_data = next((t for t in tweets_data if t.get("tweet_id") == tweet_id), None)

        if not tweet_data:
            print(f"未找到 tweet_id: {tweet_id}")
            return False

        # 获取该 tweet_id 中的句子数据
        sentences = tweet_data.get("sentences", {})

        # 如果 sentence_id 存在，替换句子文本
        if sentence_id in sentences:

            # 更新 tweet_data 中的句子文本
            tweet_data["sentences"][sentence_id]["text"] = sorted_sentence

            # 将更新后的数据写回 JSON 文件
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(tweets_data, f, ensure_ascii=False, indent=4)

            return True
        else:
            print(f"未找到 sentence_id: {sentence_id} 在 tweet_id: {tweet_id} 中")
            return False

    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return False
    except json.JSONDecodeError:
        print(f"文件内容不是有效的 JSON 格式: {file_path}")
        return False
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        return False

def construct_sorted_sentence(sentences_text):
    """
    根据句子的start_pos从小到大排序，并将对应的句子文本连接成一句话，且没有空格。

    参数:
    sentences_text: 包含所有句子的ID、文本和start_pos的列表。

    返回:
    按start_pos排序后的句子文本组成的一句话（没有空格）。
    """
    # 根据start_pos进行排序
    sorted_sentences = sorted(sentences_text, key=lambda x: x['start_pos'])

    # 将排序后的句子文本连接成一句话，没有空格
    sorted_sentence = '，'.join(sent['text'] for sent in sorted_sentences)

    return sorted_sentence

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--testing_file', type=str, default=' ', metavar='<testing_file>',
                        help='Path to testing data file')
    parser.add_argument('--output_file', type=str, default='../output/EANN/weibo_iter/', metavar='<output_file>',
                        help='Path to save results')
    parser.add_argument('--model_path', type=str, default='../models/EANN/best_model.pkl', metavar='<model_path>',
                        help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size for testing')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--event_num', type=int, default=10, help='Number of event classes')
    parser.add_argument('--text_only', type=bool, default=False, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    # 其他必要的参数...
    return parser.parse_args()

def add_adversarial_perturbation_multistep(image, sentence, model, processor, device, alpha, steps, label, vic_model, perturbed_text):
    """
    多步对抗扰动，调整图片与文本的余弦相似度，根据新闻类型减少或增加相似度。

    参数：
    - image: 输入图片。
    - sentence: 输入的文本。
    - model: CLIP 模型。
    - processor: 处理器，用于文本和图像的预处理。
    - device: 计算设备（'cuda' 或 'cpu'）。
    - alpha: 扰动的大小。
    - steps: 扰动的迭代步数。
    - label: 0 表示真新闻，1 表示假新闻。

    返回：
    - 扰动后的图片。
    """
    text_input = processor(text=[sentence], return_tensors="pt", padding=True, truncation=True).to(device)
    # 先检查是否发生截断
    '''tokenized_text = processor.tokenizer(
        perturbed_text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # 检查是否发生了截断
    if tokenized_text["input_ids"].shape[1] >= processor.tokenizer.model_max_length:
        print("Text was truncated. Using original sentence instead.")
        text_input = processor(text=[sentence], return_tensors="pt", padding=True, truncation=True).to(device)
    else:
        text_input = processor(text=[perturbed_text], return_tensors="pt", padding=True, truncation=True).to(device)
    '''
    image_tensor = image.clone().detach().to(device).requires_grad_(True)
    last_loss = None  # 用来存储最后一次的loss
    args = parse_arguments()
    # 对抗样本处理
    text, mask = process_text(perturbed_text, args)

    # 模型对抗样本预测
    text_tensor = torch.tensor(np.array(text)).unsqueeze(0).to(device)
    mask_tensor = torch.tensor(np.array(mask)).unsqueeze(0).to(device)
    for step in range(steps):
        # 预测对抗样本
        #print(f"perturbed_text: {perturbed_text}")
        #print(f"text:{text_tensor}")
        class_outputs, _ = vic_model(text_tensor, image_tensor, mask_tensor)
        prob = torch.softmax(class_outputs, dim=1)
        _, attacked_predicted = torch.max(prob, 1)
        #print(f"label:{label}, attacked_label:{attacked_predicted.item()}")
        if label != attacked_predicted.item():
            return image_tensor, last_loss
        else:
            #print(f"add noise，句子： {sentence}")
            #print(f"last_loss，last_loss： {last_loss}")

            # 提取特征
            outputs = model(**text_input, pixel_values=image_tensor)
            text_features = outputs.text_embeds
            image_features = outputs.image_embeds

            # 计算余弦相似度
            cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

            loss = cosine_similarity
            last_loss = loss.item()
            loss.backward()

            # 更新图像（沿着增大或减小损失的方向前进）
            perturbation = -alpha * image_tensor.grad.sign() if label == 0 else alpha * image_tensor.grad.sign()
            image_tensor = image_tensor + perturbation

            # 确保像素值在合法范围内
            image_tensor = torch.clamp(image_tensor, 0, 1).detach().clone().requires_grad_(True)

    '''
    # 转换为 PIL 图像
    perturbed_image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    perturbed_image = (perturbed_image * 255).astype(np.uint8)
    perturbed_image = Image.fromarray(perturbed_image)

    return perturbed_image
    '''

    return image_tensor, last_loss

def get_and_process_image(image, sentence, alpha, steps, label, vic_model, perturbed_text):
    """
    对输入的预处理图片进行对抗扰动。

    参数:
    - image: 输入图片的预处理后的Tensor变量。
    - sentence: 输入文本。
    - alpha: 扰动大小。
    - steps: 扰动步数。
    - label: 0 表示真新闻，1 表示假新闻。

    返回:
    - 扰动后的图片。
    """
    # CLIP模型加载
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # 本地模型路径
    local_model_path = "../utils/clip-vit-base-patch32"

    # 加载本地模型和处理器
    processor = CLIPProcessor.from_pretrained(local_model_path)
    model = CLIPModel.from_pretrained(local_model_path).to(device)

    # 直接传入预处理后的图片进行扰动
    perturbed_image, last_loss = add_adversarial_perturbation_multistep(image, sentence, model, processor, device, alpha, steps, label, vic_model, perturbed_text)

    return perturbed_image, last_loss

def update_similarity_value(tweet_id, attack_step, last_loss, label):
    try:
        # 打开并读取 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            tweet_data = json.load(f)

        # 查找 tweet_id 对应的数据
        tweet = next((t for t in tweet_data if t.get("tweet_id") == tweet_id), None)

        if not tweet:
            print(f"未找到 tweet_id: {tweet_id}")
            return None

        # 获取 sorted_similarities 字段
        sorted_similarities = tweet.get("sorted_similarities", [])

        # 如果 last_loss 不是 None，则执行更新
        if last_loss is not None:
            if label == 0:  # 真新闻，更新第一条
                selected_similarity = sorted_similarities[0]
            else:  # 假新闻，更新 attack_step 对应的条目
                selected_similarity = sorted_similarities[attack_step % len(sorted_similarities)]

            # 更新相似度值
            selected_similarity['similarity_value'] = last_loss

            # 按照 similarity_value 从大到小排序
            sorted_similarities = sorted(sorted_similarities, key=lambda x: x['similarity_value'], reverse=True)

            # 更新 tweet_data 中的 sorted_similarities
            tweet["sorted_similarities"] = sorted_similarities

        # 无论 last_loss 是否为 None，都将 tweet_data 更新回文件
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(tweet_data, f, ensure_ascii=False, indent=4)

        return tweet_data

    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"文件内容不是有效的 JSON 格式: {file_path}")
        return None
    except Exception as e:
        print(f"发生未知错误: {str(e)}")
        return None

def multimodal_attack(tweet_id, image, label, attack_step, vic_model, attack_history, k, alpha, steps, fasttext_model):
    """
    对给定 tweet_id 和 label 进行多模态攻击，返回扰动后的句子和扰动后的图片。

    参数：
    - tweet_id: 推文的唯一标识符。
    - label: 新闻类型标签，0 表示真新闻，1 表示假新闻。
    - attack_step: 攻击迭代次数。
    - k: 找被替换词同义词的个数。
    - alpha: 扰动的大小，控制扰动的强度。
    - steps: 扰动的迭代步数，决定扰动的精细程度。

    返回：
    - perturbed_text: 扰动后的句子。
    - perturbed_image: 扰动后的图片。
    """
    # 获取推文数据
    sample = extract_tweet_data(tweet_id, attack_step, label)
    if sample['sentence_id']:
        sentence_id = sample["sentence_id"]

        # 获取被替换分词和位置信息
        sub_word = get_word_and_pos(tweet_id, sentence_id, label, attack_history)
        input_word = sub_word['word']

        # 找到同义对抗词
        adv_words = find_adv_word(input_word, fasttext_model, k, label)
        adv_word = adv_words[0]
        adv_sentiments = adv_words[1]
        start_pos = sub_word["start_pos"]

        # 获取分句分词和位置
        sen_word = extract_words_and_positions(tweet_id, sentence_id)

        # 替换分词
        modified_sen_word = replace_word_in_sentence(sen_word, adv_word, start_pos)

        # 更改文件分词
        replace_word_in_json(tweet_id, sentence_id, adv_word, start_pos, adv_sentiments)

        # 排序并构建分句
        sorted_sent = sort_and_construct_sentence(modified_sen_word)

        # 获取推文分句及位置
        sentences = extract_sentences_text(tweet_id)

        # 替换分句
        modified_sent = replace_sentence_text(sentences, sorted_sent, sentence_id)

        # 更改文件分句
        replace_sentence_in_json_file(tweet_id, sentence_id, sorted_sent)

        # 组合分句得到完整推文
        perturbed_text = construct_sorted_sentence(modified_sent)

        # 获取图像并进行扰动
        perturbed_image, last_loss = get_and_process_image(image, sorted_sent, alpha, steps, label, vic_model, perturbed_text)
        #perturbed_image = image
        # 更新相似度值
        update_similarity_value(tweet_id, attack_step, last_loss, label)

        return perturbed_text, perturbed_image

    else:
        return None, None

'''
tweet_id = "3906158693319838"
label = 1  # 假新闻
alpha = 0.01  # 调整扰动大小
steps = 10
k = 3
# 示例：如何预处理输入图片并传入
def preprocess_image(image_path):
    """
    预处理图片，并返回处理后的 Tensor。
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = data_transforms(image).unsqueeze(0)  # Add batch dimension
    return image_tensor
image = preprocess_image('../data/weibo/nonrumor_images/62aad664jw1exr575cusqj20u01hc7fr.jpg')

for step in range(3):  # 调用 3 次
    sorted_sentence, perturbed_image = multimodal_attack(
        tweet_id, image, label, step, k, alpha, steps, fasttext_model
    )
    image = perturbed_image
    print(f"攻击 {step+1} 结果:")

    if sorted_sentence:
        print("排序后的句子:", sorted_sentence)
    if perturbed_image is not None:
        perturbed_image = perturbed_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        perturbed_image = (perturbed_image * 255).astype(np.uint8)
        perturbed_image = Image.fromarray(perturbed_image)
        plt.imshow(perturbed_image)
        plt.axis("off")
        plt.show()
'''