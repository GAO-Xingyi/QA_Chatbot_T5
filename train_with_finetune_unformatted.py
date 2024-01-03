import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
from torch.utils.data import Dataset, DataLoader

class T5PegasusTokenizer(T5Tokenizer):
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

class QADataset(Dataset):
    def __init__(self, passages):
        self.passages = passages
        self.tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        passage = self.passages[idx]

        # Only include text, no need to construct a question
        input_text = f"{passage}"

        # target_text can be an empty string, as we don't need a target sequence
        target_text = ""

        # Ensure the same max_length for both input and target
        max_length = 1024

        input_ids = self.tokenizer.encode(
            input_text, return_tensors="pt", max_length=max_length, truncation=True, padding=True
        )
        target_ids = self.tokenizer.encode(
            target_text, return_tensors="pt", max_length=max_length, truncation=True, padding=True
        )

        return {"input_ids": input_ids, "target_ids": target_ids}

def init_argument():
    parser = argparse.ArgumentParser(description='unformatted_QA')
    parser.add_argument('--pretrain_model', default='t5-pegasus')
    parser.add_argument('--num_epoch', default=200, help='number of epoch')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = init_argument()

    # 读取知识库文本
    file_path = "data/unformatted.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        passages = file.readlines()

    # 创建数据集
    dataset = QADataset(passages)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 加载T5模型
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 模型训练
    num_epochs = args.num_epoch
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]

            optimizer.zero_grad()
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    # 保存训练好的模型
    model.save_pretrained("zero_shot_QA_model")


