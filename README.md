# 项目介绍

## 中文QA生成与续训练

中文QA生成与续训练项目结合了中文QA生成和模型续训练的功能。通过使用T5预训练模型、中文分词技术以及BERT等先进技术，我们实现了一个能够生成中文问答对的模型，并进一步加入续训练的能力，以提高模型的性能。

## 功能特性

1. **数据加载与处理：** 项目支持灵活的数据加载，用户可自定义问题与回答的格式，方便适应不同的应用场景。
2. **T5预训练模型微调：** 利用T5预训练模型，通过微调实现对中文问答生成任务的训练。
3. **模型续训练：** 提供续训练功能，用户可在已有模型的基础上进行进一步训练，以适应新的数据和任务。
4. **BLEU分数计算：** 项目包含计算BLEU分数的功能，用于评估生成的中文回答与实际回答之间的相似度。

## 用法

1. **初始化参数：** 使用`init_argument`函数初始化训练所需的参数，包括训练数据路径、预训练模型路径等。
2. **数据准备：** 调用`prepare_data`函数准备训练数据和验证数据。
3. **模型初始化与训练：** 使用MT5ForConditionalGeneration模型进行初始化，然后调用`continue_training`函数进行续训练。
4. **结果评估：** 训练完成后，可通过BLEU分数等指标评估模型性能。

## 示例

```python
# 初始化参数
args = init_argument()

# 准备训练数据和验证数据
tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
train_data = prepare_data(args, args.train_data, tokenizer, term='train')
dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')

# 初始化模型和优化器
model = MT5ForConditionalGeneration.from_pretrained(args.pretrain_model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# 续训练
continue_training(model, train_data, num_epochs=args.num_epoch, learning_rate=args.lr)
```

## 依赖项

- Python 3.x
- PyTorch
- transformers
- bert4torch
- nltk

## 注意事项

- 请根据实际需求调整参数，如续训练的轮数、学习率等。
- 请确保安装了所需的Python库，可以通过`pip install torch transformers nltk bert4torch`安装。
- 需要提供实际的预训练模型路径和训练数据路径。

这个项目旨在为中文问答生成任务提供一个灵活而强大的工具，同时通过续训练功能，使模型能够持续学习并适应不断变化的需求。