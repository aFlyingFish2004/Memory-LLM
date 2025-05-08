import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
from utils.memory_access import process_query

# 模型路径 & DB
model_path = "models/Qwen2.5-1.5B-Read"
db_path = "memory/memory.db"

# What is the relationship between Pasay City and the Philippines?
# query = '{"name": "MEM_READ", "arguments": {"subject": "Pasay City", "object": "Philippines"}}'
# print(process_query(query, db_path))

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()

# 系统提示词
system_prompt = (
    "You are a memory-augmented assistant. When you need information to answer a question, "
    "first generate a MEM_READ API call to retrieve relevant memory triplets. "
    "If the memory retrieval result is empty, respond truthfully by saying you don't know."
)

def build_prompt(history, system_prompt):
    """根据已有 history（含最后一次 assistant 条目）和系统提示，构建 prompt 到 <|im_start|>assistant"""
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for user, assistant in history:
        prompt += f"<|im_start|>user\n{user}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    # 末尾留给模型生成
    prompt += "<|im_start|>assistant\n"
    return prompt

history = []
print(f"模型当前加载到的设备是: {model.device}")

while True:
    print("==========用户==========")
    # user_input = input("你：")
    user_input = input()
    if user_input.strip().lower() in ["exit", "quit", "q"]:
        break

    # 第一阶段：模型生成可能包含 tool_call
    # 构建 prompt（带 user）
    stage1_prompt = build_prompt(history + [(user_input, "")], system_prompt).replace(
        "<|im_start|>assistant\n", f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(stage1_prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = Thread(target=model.generate, kwargs={
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 1024,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.7,
        "repetition_penalty": 1.1
    })
    thread.start()

    print("==========助手==========")
    # print("助手：", end="", flush=True)
    raw = ""
    for chunk in streamer:
        print(chunk, end="", flush=True)
        raw += chunk
    print()

    # 如果有 tool_call，就执行并进入第二阶段
    if "<tool_call>" in raw:
        # 解析 and 调用 process_query
        try:
            tool_result = process_query(raw, db_path)
            print("==========工具调用结果==========")
            # print(f"工具调用结果：{tool_result}")
            print(tool_result)
        except Exception as e:
            print(f"[Error] 解析 tool_call 失败: {e}")
            tool_result = "Error: invalid tool_call format"

        # 存入 history：第一条 assistant 是 raw（包含 tool_call）
        history.append((user_input, raw))

        # 第二阶段：模型基于 tool_result 继续回答
        history.append((None, tool_result))  # 用 None 标记这是工具结果，而非用户
        stage2_prompt = build_prompt(history, system_prompt)
        # 注意：build_prompt 会忽略 None 用户，因此我们临时在拼接时跳过 None 的 user
        # 实际实现可调整 build_prompt 以跳过 None

        # 准备输入
        inputs2 = tokenizer(stage2_prompt, return_tensors="pt").to(model.device)
        streamer2 = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread2 = Thread(target=model.generate, kwargs={
            **inputs2,
            "streamer": streamer2,
            "max_new_tokens": 512,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.6,
            "repetition_penalty": 1.0
        })
        thread2.start()

        print("==========助手==========")
        # print("助手：", end="", flush=True)
        final_resp = ""
        for chunk in streamer2:
            print(chunk, end="", flush=True)
            final_resp += chunk
        print()

        # 把最终回答存入 history（替换掉 None 标记行的内容）
        history[-1] = (tool_result, final_resp)

    else:
        # 不含 tool_call 就正常对话
        history.append((user_input, raw))

    # print('==========history==========')
    # print(history)
    # print('===========================')
