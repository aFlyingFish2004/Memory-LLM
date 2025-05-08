import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import gradio as gr

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

# 系统提示词
system_prompt = "You are a memory-augmented assistant. When you need information to answer a question, first generate a MEM_READ API call to retrieve relevant memory triplets. If the memory retrieval result is empty, respond truthfully by saying you don't know."

# 构建 prompt
def build_prompt(history, current_input):
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for turn in history:
        user = turn["content"] if turn["role"] == "user" else ""
        assistant = turn["content"] if turn["role"] == "assistant" else ""
        if user:
            prompt += f"<|im_start|>user\n{user}<|im_end|>\n"
        if assistant:
            prompt += f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{current_input}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

# 推理函数（返回 OpenAI 风格字典）
def predict(user_input, history, top_p, temperature):
    history = history or []

    prompt = build_prompt(history, user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=1.1
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_output = ""
    new_history = history + [{"role": "user", "content": user_input}]
    for new_text in streamer:
        partial_output += new_text
        yield new_history + [{"role": "assistant", "content": partial_output}]

    # 最终更新完整历史
    return new_history + [{"role": "assistant", "content": partial_output}]


# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 💬 本地大模型对话界面（Qwen2.5）")

    chatbot = gr.Chatbot(type="messages")
    with gr.Row():
        with gr.Column(scale=8):
            user_input = gr.Textbox(placeholder="请输入您的问题", show_label=False)
        with gr.Column(scale=1):
            submit_btn = gr.Button("提交")

    with gr.Accordion("参数设置", open=False):
        top_p = gr.Slider(minimum=0.01, maximum=1.0, value=0.95, label="Top-p 采样值")
        temperature = gr.Slider(minimum=0.01, maximum=1.5, value=0.95, label="温度系数")

    clear_btn = gr.Button("清空历史")
    history_state = gr.State([])

    # 提交事件绑定
    submit_btn.click(
        fn=predict,
        inputs=[user_input, history_state, top_p, temperature],
        outputs=[chatbot],
        show_progress=True
    ).then(
        lambda chat, _: (None, chat),
        inputs=[history_state, chatbot],
        outputs=[user_input, history_state]
    )

    user_input.submit(
        fn=predict,
        inputs=[user_input, history_state, top_p, temperature],
        outputs=[chatbot]
    ).then(
        lambda chat, _: (None, chat),
        inputs=[history_state, chatbot],
        outputs=[user_input, history_state]
    )

    clear_btn.click(lambda: ([], []), inputs=[], outputs=[chatbot, history_state])

# 启动 Gradio 服务，避免端口冲突
demo.launch(server_port=7861)
