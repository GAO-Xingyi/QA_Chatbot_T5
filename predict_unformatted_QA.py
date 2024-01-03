def generate_answer(question, model, tokenizer):
    input_text = f"answer question: {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    answer_ids = model.generate(input_ids)
    answer_text = tokenizer.decode(answer_ids[0], skip_special_tokens=True)

    return answer_text

# 示例问题
user_question = "什么是长篇文本?"

# 加载训练好的模型
trained_model = T5ForConditionalGeneration.from_pretrained("t5_qa_model")
trained_tokenizer = T5Tokenizer.from_pretrained("t5_qa_model")

# 生成回答
generated_answer = generate_answer(user_question, trained_model, trained_tokenizer)

# 打印生成的回答
print("Generated Answer:")
print(generated_answer)
