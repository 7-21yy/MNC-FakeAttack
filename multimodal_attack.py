import json
import os
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from gensim.models import FastText
from snownlp import SnowNLP
from transformers import CLIPProcessor, CLIPModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#fasttext_model = FastText.load_fasttext_format('../utils/cc.zh.300.bin')

file_path = "../data/weibo/tweets/updated_merged_tweet_with_similarities_filtered.json"

def extract_tweet_data(tweet_id):
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
                "image_content": None
            }

        # 检查 sorted_similarities
        sorted_similarities = tweet_data.get("sorted_similarities", [])
        if sorted_similarities:
            first_similarity = sorted_similarities[0]
            sentence_id = first_similarity.get("sentence_id")
            image_id = first_similarity.get("image_id")

            sentence_content = tweet_data.get("sentences", {}).get(sentence_id, {}).get("text")
            image_content = tweet_data.get("images", {}).get(image_id)

            return {
                "sentence_id": sentence_id,
                "sentence_content": sentence_content,
                "image_id": image_id,
                "image_content": image_content
            }

        # sorted_similarities 为空时，取第一条句子和图片
        sentences = tweet_data.get("sentences", {})
        images = tweet_data.get("images", {})

        # 获取第一个句子ID和内容
        sentence_id, sentence_data = next(iter(sentences.items()), (None, {}))
        sentence_content = sentence_data.get("text")

        # 获取第一个图片ID和内容
        image_id, image_content = next(iter(images.items()), (None, None))

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
            "image_content": None
        }

def get_word_and_pos(tweet_id, sentence_id, label):
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

        if label == 0:  # 如果是“真新闻” (label = 0)
            word = tokenized_data[0].get("word")
            start_pos = tokenized_data[0].get("start_pos")
            return {
                "word": word,
                "start_pos": start_pos,
            }
        else:  # 如果是“假新闻” (label = 1)
            # 遍历 tokenized_data，查找第一个词性不相同的词
            for i in range(1, len(tokenized_data)):  # 从第二个词开始
                if tokenized_data[i]["pos_tag"] != tokenized_data[i - 1]["pos_tag"]:
                    # 找到与前一个词性不同的词，返回该词及其起始位置
                    word = tokenized_data[i - 1].get("word")
                    start_pos = tokenized_data[i - 1].get("start_pos")
                    return {
                        "word": word,
                        "start_pos": start_pos,
                    }

            # 如果所有词的词性都相同，返回最后一个词
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
    s = SnowNLP(word)
    return s.sentiments  # 返回情感得分，范围 [0, 1]

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
    neighbors = model.wv.most_similar(word, topn=k)

    # 过滤出同义词并计算情感分数
    word_sentiments = []
    for similar_word, similarity in neighbors:
        sentiment_score = get_sentiment_score(similar_word)
        word_sentiments.append((similar_word, sentiment_score))

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
def add_adversarial_perturbation_multistep(image, sentence, model, processor, device, alpha, steps, label):
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
    image_tensor = image.to(device)  # image is now preprocessed and passed directly as a tensor
    image_tensor.requires_grad = True

    for step in range(steps):
        # 提取特征
        outputs = model(**text_input, pixel_values=image_tensor)
        text_features = outputs.text_embeds
        image_features = outputs.image_embeds

        # 计算余弦相似度
        cosine_similarity = torch.nn.functional.cosine_similarity(image_features, text_features)

        loss = cosine_similarity
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

    return image_tensor

def get_and_process_image(image, sentence, alpha, steps, label):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 本地模型路径
    local_model_path = "../utils/clip-vit-base-patch32"

    # 加载本地模型和处理器
    processor = CLIPProcessor.from_pretrained(local_model_path)
    model = CLIPModel.from_pretrained(local_model_path).to(device)

    # 直接传入预处理后的图片进行扰动
    perturbed_image = add_adversarial_perturbation_multistep(image, sentence, model, processor, device, alpha, steps, label)

    return perturbed_image

def multimodal_attack(tweet_id, image, label, k, alpha, steps, fasttext_model):
    """
    对给定 tweet_id 和 label 进行多模态攻击，返回扰动后的句子和扰动后的图片。

    参数：
    - tweet_id: 推文的唯一标识符。
    - label: 新闻类型标签，0 表示真新闻，1 表示假新闻。
    - iter: 攻击迭代次数。
    - k: 找被替换词同义词的个数。
    - alpha: 扰动的大小，控制扰动的强度。
    - steps: 扰动的迭代步数，决定扰动的精细程度。

    返回：
    - perturbed_text: 扰动后的句子。
    - perturbed_image: 扰动后的图片。
    """
    # 获取推文数据
    sample = extract_tweet_data(tweet_id)
    if sample['sentence_id']:
        sentence_id = sample["sentence_id"]

        # 获取被替换分词和位置信息
        sub_word = get_word_and_pos(tweet_id, sentence_id, label)
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

        # 排序并构建分句
        sorted_sent = sort_and_construct_sentence(modified_sen_word)

        # 获取推文分句及位置
        sentences = extract_sentences_text(tweet_id)

        # 替换分句
        modified_sent = replace_sentence_text(sentences, sorted_sent, sentence_id)

        # 组合分句得到完整推文
        perturbed_text = construct_sorted_sentence(modified_sent)

        # 获取图像并进行扰动
        perturbed_image = get_and_process_image(image, perturbed_text, alpha, steps, label)

        return perturbed_text, perturbed_image

    else:
        return None, None

'''
tweet_id = "3875209226459010"
label = 1  # 假新闻
alpha = 0.01  # 调整扰动大小
steps = 10
k = 5

sorted_sentence, perturbed_image = multimodal_attack(tweet_id, label, k, alpha, steps, fasttext_model)

if sorted_sentence:
    print("排序后的句子:", sorted_sentence)
if perturbed_image:
    plt.imshow(perturbed_image)
    plt.axis("off")
    plt.show()
'''