import os

os.system(
    'pip install modelscope==1.4.3 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html'
)
os.system('pip install gradio==3.23.0 -U')
import gradio as gr
# 使用
import torch
from modelscope.models.nlp import T5ForConditionalGeneration
# 加载模型
from modelscope.pipelines import pipeline
from modelscope.preprocessors import TextGenerationT5Preprocessor
from modelscope.utils.constant import Tasks

# ChatGLM-6B

chatglm_pipe = pipeline(task=Tasks.chat,
                        model='ZhipuAI/ChatGLM-6B',
                        model_revision='v1.0.7')


def chatglm_predict(input, history=None):
    if history is None:
        history = []
    inputs = inputs = {'text': input, 'history': history}
    result = chatglm_pipe(inputs)
    history = result['history']

    return history, history

# ChatYuan

model = T5ForConditionalGeneration.from_pretrained('ClueAI/ChatYuan-large-v2',
                                                   revision='v1.0.0')
preprocessor = TextGenerationT5Preprocessor(model.model_dir)
pipeline_t2t = pipeline(task=Tasks.text2text_generation,
                        model=model,
                        preprocessor=preprocessor)


def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text


def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t").replace('%20', '  ')



def answer(text, sample=True, top_p=1, temperature=0.7):
    '''sample：是否抽样。生成任务，可以设置为True;
  top_p：0-1之间，生成的内容越多样'''
    text = preprocess(text)

    if not sample:
        out_text = pipeline_t2t(text,
                                return_dict_in_generate=True,
                                output_scores=False,
                                max_new_tokens=1024,
                                num_beams=1,
                                length_penalty=0.6)
    else:
        out_text = pipeline_t2t(text,
                                return_dict_in_generate=True,
                                output_scores=False,
                                max_new_tokens=1024,
                                do_sample=True,
                                top_p=top_p,
                                temperature=temperature,
                                no_repeat_ngram_size=3)

    return postprocess(out_text["text"])


def clear_session():
    return '', None, '', None, '', None


def chatyuan(input, history):
    history = history or []
    if len(history) > 5:
        history = history[-5:]

    context = "\n".join([
        f"用户：{input_text}\n小元：{answer_text}"
        for input_text, answer_text in history
    ])

    input_text = context + "\n用户：" + input + "\n小元："
    output_text = answer(input_text)
    history.append((input, output_text))

    return history, history



# Minimax

def minimax_predict(input, history):
    history = history or []
    if len(history) > 5:
        history = history[-5:]
    import requests

    group_id = os.getenv('group_id')

    api_key1 = os.getenv('api_key1')
    api_key2 = os.getenv('api_key2')
    api_key = api_key1 + api_key2

    url = f'https://api.minimax.chat/v1/text/chatcompletion?GroupId={group_id}'
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    request_body = {
        "model": "abab5-chat",
        "tokens_to_generate": 512,
        'messages': []
    }

    for i in history:
        prompt = i[0]
        reply = i[1]
        request_body['messages'].append({
            "sender_type": "USER",
            "text": prompt
        })
        request_body['messages'].append({"sender_type": "BOT", "text": reply})

    request_body['messages'].append({"sender_type": "USER", "text": input})
    response = requests.post(url, headers=headers, json=request_body)
    reply = response.json()['reply']
    #  将当次的ai回复内容加入messages
    request_body['messages'].append({"sender_type": "BOT", "text": reply})
    history.append((input, reply))
    return history, history


def predict(input, history1, history2, history3):

    history1, history1 = chatyuan(input, history1)

    history2, history2 = chatglm_predict(input, history2)

    history3, history3 = minimax_predict(input, history3)

    print(history1, history2, history3)
    return '', history1, history1, history2, history2, history3, history3


block = gr.Blocks()

with block as demo:
    gr.Markdown("""<h1><center>LLM-EVALUATION</center></h1>
    """)
    with gr.Row():

        chatbot1 = gr.Chatbot(label='ChatYuan-large-v2')
        chatbot2 = gr.Chatbot(label='ChatGLM-6B')
        chatbot3 = gr.Chatbot(label='MiniMax')
        state1 = gr.State()
        state2 = gr.State()
        state3 = gr.State()
    message = gr.Textbox()

    message.submit(predict,
                   inputs=[message, state1, state2, state3],
                   outputs=[
                       message, chatbot1, state1, chatbot2, state2, chatbot3,
                       state3
                   ])
    with gr.Row():
        clear_history = gr.Button("🧹 清除历史对话")
        send = gr.Button("🚀 发送")

    send.click(predict,
               inputs=[message, state1, state2, state3],
               outputs=[
                   message, chatbot1, state1, chatbot2, state2, chatbot3,
                   state3
               ])
    clear_history.click(
        fn=clear_session,
        inputs=[],
        outputs=[chatbot1, state1, chatbot2, state2, chatbot3, state3],
        queue=False)

if __name__ == "__main__":
    demo.queue().launch(share=False)