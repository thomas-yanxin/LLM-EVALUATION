#该应用创建工具共包含三个区域，顶部工具栏，左侧代码区，右侧交互效果区，其中右侧交互效果是通过左侧代码生成的，存在对照关系。
#顶部工具栏：运行、保存、新开浏览器打开、实时预览开关，针对运行和在浏览器打开选项进行重要说明：
#[运行]：交互效果并非实时更新，代码变更后，需点击运行按钮获得最新交互效果。
#[在浏览器打开]：新建页面查看交互效果。
#以下为应用创建工具的示例代码

import os

os.system("python -m pip install paddlepaddle-gpu==0.0.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html")
os.system("pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html")
import gradio as gr
# 使用
from chatglm_paddle import ChatGLM_Predictor
from chatyuan_paddle import ChatYuan_Predictor

chatyuan_predictor = ChatYuan_Predictor()
chatglm_predictor = ChatGLM_Predictor()

batch_size = 1

def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts

def clear_session():
    return '', None, '', None


def chatglm(input, history):
    print(input)
    history = history or []
    if len(history) > 5:
        history = history[-5:]
    all_texts = [text for _, text in history]
    all_texts.append(input)
    print(all_texts)
    batch_texts = batchfy_text(all_texts, batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = chatglm_predictor.predict(texts)
    print(outputs["result"][0])
    history.append((input, outputs["result"][0]))

    return history, history    
    

def chatyuan(input, history):
    history = history or []
    if len(history) > 5:
        history = history[-5:]

    context = "\n".join([
        f"用户：{input_text}\n小元：{answer_text}"
        for input_text, answer_text in history
    ])

    input_text = context + "\n用户：" + input + "\n小元："
    output_text = chatyuan_predictor.chatyuan_answer(input_text)
    history.append((input, output_text))

    return history, history


def predict(input, history1, history2):

    history1, history1 = chatyuan(input, history1)

    history2, history2 = chatglm(input, history2)

    print(history1, history2)

    return '', history1, history1, history2, history2


block = gr.Blocks()

with block as demo:
    gr.Markdown("""<h1><center>LLM-EVALUATION</center></h1>

<font size=4>
👏 &nbsp; 本项目提供一键式的基于多个LLM的生成效果评测。方便开发者从LLM的生成效果角度自我评价模型效果，也方便高阶开发者更为直观且准确地分析模型在场景、参数等之间的差异。

👀 &nbsp; 目前已支持ChatYuan-large-v2、ChatGLM-6B等模型，后续将集成生态内更多模型，致力于探索中文领域内的以用户体验为基础的LLM测评机制。目前仍是雏形，欢迎共建：[Github地址](https://github.com/thomas-yanxin/LLM-EVALUATION)

</font>
    """)
    with gr.Row():

        chatbot1 = gr.Chatbot(label='ChatYuan-large-v2')
        chatbot2 = gr.Chatbot(label='ChatGLM-6B')

        state1 = gr.State()
        state2 = gr.State()

    message = gr.Textbox(label='可以在这里多多提问嗷！')

    message.submit(predict,
                   inputs=[message, state1, state2],
                   outputs=[
                       message, chatbot1, state1, chatbot2, state2, 
                   ])
    with gr.Row():
        clear_history = gr.Button("🧹 清除历史对话")
        send = gr.Button("🚀 发送")

    send.click(predict,
               inputs=[message, state1, state2],
               outputs=[
                   message, chatbot1, state1, chatbot2, state2
               ])
    clear_history.click(
        fn=clear_session,
        inputs=[],
        outputs=[chatbot1, state1, chatbot2, state2],
        queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=True)

