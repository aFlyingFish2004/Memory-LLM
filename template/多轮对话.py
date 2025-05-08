import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

# 模型路径
model_path = "models/Qwen2.5-1.5B-Read"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

# Qwen 模板：多轮对话包装
system_prompt = "You are a memory-augmented assistant. When you need information to answer a question, first generate a MEM_READ API call to retrieve relevant memory triplets. If the memory retrieval result is empty, respond truthfully by saying you don't know."

def build_prompt(history, current_input, system_prompt):
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for user, assistant in history:
        prompt += f"<|im_start|>user\n{user}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{current_input}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


# 多轮对话历史记录
history = []

# 主循环
while True:
    user_input = input("你：")
    if user_input.strip().lower() in ["exit", "quit", "q"]:
        break

    # 构建 prompt
    prompt = build_prompt(history, user_input, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 设置流式输出
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.1
    )

    # 用线程非阻塞地生成
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("助手：", end="", flush=True)
    outputs = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        outputs += new_text
    print()

    # 更新历史
    history.append((user_input, outputs))