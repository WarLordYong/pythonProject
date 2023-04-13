import os
import paddle
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import MapDataset
from paddle.dataset.common import md5file
from paddlenlp.datasets import DatasetBuilder
from paddlenlp.taskflow.dependency_parsing import convert_example
import pandas as pd
import paddle.nn.functional as F
import numpy as np

# train1 = pd.read_csv('virus_train.csv')  # 疫情微博训练数据集
# train2 = pd.read_csv('usual_train.csv')  # 通用微博训练数据集
# train = pd.read_csv('train.csv')  # 拼接训练集


label_map= {0: 'happy', 1: 'sad', 2: 'neutral', 3: 'fear', 4: 'angry', 5: 'surprise'}


model = ppnlp.transformers.NeZhaForSequenceClassification.from_pretrained('nezha-large-wwm-chinese', num_classes=6)
params_path = 'checkpoint/model_state.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)


# 定义数据加载和处理函数
def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    qtconcat = example["text_a"]
    encoded_inputs = tokenizer(text=qtconcat, max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

def predict(model, data, tokenizer, label_map, batch_size=1):
    examples = []
    for text in data:
        input_ids, segment_ids = convert_example(
            text,
            tokenizer,
            max_seq_length=128,
            is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
    ): fn(samples)

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


# 定义要进行预测的样本数据
data = [
    # angry
    {"text_a": '更年期的女boss真的让人受不了，烦躁'},
    {"text_a":"起的太晚 怪你"},
    # fear
    {"text_a":'尼玛吓死我了，人家剪个头发回来跟劳改犯一样短的可怕，后面什么鬼[黑线][黑线][黑线][白眼][白眼]'},
    # sad
    {"text_a":'一个人真无聊，美食都没味了，你要在就好了…唉………'},
    # neutral
    {"text_a":"这个村的年轻人大多数都出外打工。"},

    # happy
    {"text_a":"谢谢honey们帮我庆祝生日！！！谢谢你们的祝福，谢谢身边的所有人！爱你们"},
    # surprise
    {"text_a":"我竟然才知道我有一个富二代加官二代加红二代的朋友"},

]
tokenizer = ppnlp.transformers.NeZhaTokenizer.from_pretrained('nezha-large-wwm-chinese',cache_dir="./checkpoint/")

results = predict(model, data, tokenizer, label_map, batch_size=1)

for idx, text in enumerate(data):
    print('Data: {} \t Lable: {}'.format(text['text_a'], results[idx]))