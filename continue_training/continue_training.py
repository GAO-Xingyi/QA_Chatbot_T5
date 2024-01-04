import os
import re
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm
from bert4torch.models import *
from torch.utils.data import DataLoader, Dataset
from transformers import MT5ForConditionalGeneration, BertTokenizer
from transformers import T5ForConditionalGeneration, AdamW
#from transformers import BertTokenizer
import jieba
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk
from tqdm import tqdm

int_classes = int
string_classes = str

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

import sys

sys.setrecursionlimit(10**6)


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cur = l.strip().split('\t')
            try:
                if len(cur) == 2:
                    question, answer = cur[0], cur[1]
                    D.append({'question': question, 'answer': answer})
                elif len(cur) == 1:
                    answer = cur[0]
                    D.append({'answer': answer})
            except ValueError:
                print("Invalid data format: {}".format(cur))
    return D


def create_data(data, tokenizer, max_len_generate, term='train'):
    ret, flag = [], True
    for item in data:
        question = item.get('question', 'Default Question')
        answer = item['answer']

        question_ids = tokenizer.encode(
            question, max_length=max_len_generate, truncation='only_first')
        answer_ids = tokenizer.encode(
            answer, max_length=max_len_generate, truncation='only_first')

        if flag and term == 'train':
            flag = False
            print(question)
        if term == 'train':
            features = {'input_ids': question_ids,
                        'decoder_input_ids': answer_ids,
                        'attention_mask': [1] * len(question_ids),
                        'decoder_attention_mask': [1] * len(answer_ids),
                        'answer': answer
                        }
        elif term == 'dev':
            features = {'input_ids': question_ids,
                        'attention_mask': [1] * len(question_ids),
                        'answer': answer
                        }
        ret.append(features)
    return ret


class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """

    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    # 使用 torch.cat 替代 torch.stack
    if isinstance(batch[0], torch.Tensor):
        return torch.cat([x.view(1, *x.shape) for x in batch], dim=0)

    elem = batch[0]
    elem_type = type(elem)
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))
            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples)
                           for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)
        return default_collate([default_collate(elem) for elem in batch])
    elif isinstance(elem, torch.Tensor) and elem.numel() > 0:
        return elem.resize_(0)
    raise TypeError(default_collate_err_msg_format.format(elem_type))


def prepare_data(args, data_path, tokenizer, term='train'):
    """准备batch数据
    """
    data = load_data(data_path)
    data = create_data(data, tokenizer, args.max_len_generate, term)
    data = KeyDataset(data)
    data = DataLoader(data, batch_size=args.batch_size,
                      collate_fn=default_collate)
    return data.dataset  # 返回 DataLoader 对象的 dataset 属性


def compute_bleu(references, hypotheses):
    """
    计算BLEU分数
    """
    smooth = nltk.translate.bleu_score.SmoothingFunction()
    bleu_score = nltk.translate.bleu_score.corpus_bleu(references, hypotheses, smoothing_function=smooth.method1)
    return bleu_score




def train_model(model, adam, train_data, dev_data, tokenizer, device, args):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    best_bleu_score = 0
    for epoch in range(args.num_epoch):
        model.train()
        for i, cur in enumerate(tqdm(train_data, desc='Epoch {}:'.format(epoch))):
            cur = {k: v.to(device) for k, v in cur.items()}
            prob = model(**cur)[0]
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            if i % 100 == 0:
                print("Iter {}:  Training Loss: {}".format(i, loss.item()))
            loss.backward()
            adam.step()
            adam.zero_grad()

        # 保存模型
        if args.data_parallel and torch.cuda.is_available():
            torch.save(model.module, os.path.join(args.model_dir, 'summary_model'))
        else:
            torch.save(model, os.path.join(args.model_dir, 'summary_model'))

    # 验证
    model.eval()
    generated_responses = []
    reference_answers = []
    for feature in tqdm(dev_data):
        title = feature['answer']
        content = {k: v.to(device) for k, v in feature.items() if k != 'answer'}
        if args.data_parallel and torch.cuda.is_available():
            gen = model.module.generate(max_length=args.max_len_generate,
                                        eos_token_id=tokenizer.sep_token_id,
                                        decoder_start_token_id=tokenizer.cls_token_id,
                                        **content)
        else:
            gen = model.generate(max_length=args.max_len_generate,
                                 eos_token_id=tokenizer.sep_token_id,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 **content)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
        gen = [item.replace(' ', '') for item in gen]
        generated_responses.extend(gen)
        reference_answers.extend([title])

    # 计算BLEU分数
    bleu_score = compute_bleu([reference_answers], [generated_responses])
    print("Validation BLEU Score: {}".format(bleu_score))



def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--train_data', default='./update_dataset/caoling.tsv')
    parser.add_argument('--dev_data', default='./update_dataset/caoling.tsv')
    parser.add_argument('--pretrain_model', default='../t5_pegasus_pretrain')
    parser.add_argument('--model_dir', default='./continue_training_model')

    parser.add_argument('--num_epoch', default=1000, help='number of epoch')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    parser.add_argument('--max_len', default=25, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=350, help='max length of outputs')

    args = parser.parse_args()
    return args


def continue_training(model, optimizer, train_data, dev_data, tokenizer, device, args):
    # 将模型设置为训练模式
    model.train()

    best_bleu_score = 0
    for epoch in range(args.num_epoch):
        # 初始化损失
        total_loss = 0.0

        # 使用tqdm创建进度条
        for cur in tqdm(train_data, desc=f'Epoch {epoch + 1}/{args.num_epoch}'):
            # 将数据移到设备上
            print(type(cur))
            print(cur)
            cur.pop('answer')
            cur = {k: torch.tensor(v) for k, v in cur.items()}
            print(cur.len())
            print(cur.data())
            # 前向传播
            prob = model(**cur)[0]
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 累积总损失
            total_loss += loss.item()

        # 打印每个epoch的平均损失
        average_loss = total_loss / len(train_data)
        print(f"Epoch {epoch + 1}/{args.num_epoch}, 平均损失: {average_loss}")

        # 验证
        model.eval()
        generated_responses = []
        reference_answers = []
        for feature in tqdm(dev_data):
            title = feature['answer']
            content = {k: v for k, v in feature.items() if k != 'answer'}
            gen = model.generate(max_length=args.max_len_generate,
                                        eos_token_id=tokenizer.sep_token_id,
                                        decoder_start_token_id=tokenizer.cls_token_id,
                                        **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            generated_responses.extend(gen)
            reference_answers.extend([title])

        # 计算BLEU分数
        bleu_score = compute_bleu([reference_answers], [generated_responses])
        print("Validation BLEU Score: {}".format(bleu_score))

        # 如果当前BLEU分数高于之前最好的分数，则保存模型
        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            torch.save(model, os.path.join(args.model_dir, 'summary_model'))

    # 保存最终模型
    torch.save(model, os.path.join(args.model_dir, 'summary_model_final'))





if __name__ == '__main__':
    args = init_argument()

    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    train_data = prepare_data(args, args.train_data, tokenizer, term='train')
    dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')
    print(type(train_data))
    print(type(dev_data))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化模型和优化器
    model = MT5ForConditionalGeneration.from_pretrained(args.pretrain_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 继续训练
    # continue_training(model, train_data,tokenizer=tokenizer, num_epochs=args.num_epoch, learning_rate=args.lr)
    continue_training(model,optimizer,train_data,dev_data, tokenizer, device, args)
