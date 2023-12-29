# 读取包含问答对的 .txt 文件，将其内容保存为 .tsv 文件

input_file_path = '/root/gpt/seq2seq/data/caoling.txt'
output_file_path = '/root/gpt/seq2seq/data/caoling.tsv'

with open(input_file_path, 'r', encoding='utf-8') as infile:
    lines = infile.read().split('\n')  # 通过换行符分割问答对

# 将问答对分隔并保存为 .tsv 文件
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for line in lines:
        # 跳过空行
        if not line:
            continue

        # 使用制表符 '\t' 分隔问题和答案
        parts = line.strip().split('\t')
        
        # 检查是否包含预期的两个元素
        if len(parts) == 2:
            question, answer = parts
            
            # 将问答对写入 .tsv 文件
            outfile.write(f'{question}\t{answer}\n')
        else:
            print(f"Warning: Skipped line with unexpected format: {line}")
