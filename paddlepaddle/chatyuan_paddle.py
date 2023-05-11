from paddlenlp.transformers import AutoTokenizer, T5ForConditionalGeneration


class ChatYuan_Predictor():
    
    def __init__(self):
        self.ChatYuan_tokenizer = AutoTokenizer.from_pretrained("ClueAI/ChatYuan-large-v2", from_hf_hub=False)
        self.ChatYuan_model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2", from_hf_hub=False)

        self.ChatYuan_model.eval()
    # 输入前处理
    def preprocess(self, text):
        text = text.replace("\n", "\\n").replace("\t", "\\t")
        return text
    # 输出后处理 
    def postprocess(self,text):
        return text.replace("\\n", "\n").replace("\\t", "\t")
    # 模型回答
    def chatyuan_answer(self, text, sample=True, top_p=1, temperature=0.7):
        '''sample：是否抽样。生成任务，可以设置为True;
        top_p：0-1之间，生成的内容越多样'''
        text = self.preprocess(text)
        # 编码输入
        encoding = self.ChatYuan_tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pd")
        # 生成
        if not sample:
            out = self.ChatYuan_model.generate(**encoding,
                                return_dict_in_generate=True, 
                                output_scores=False, 
                                max_length=512,
                                max_new_tokens=2048, 
                                num_beams=1, 
                                length_penalty=0.6,
                                no_repeat_ngram_size=12,
                                top_p=top_p, 
                                temperature=temperature
                                )
        else:
            out = self.ChatYuan_model.generate(**encoding, 
                                return_dict_in_generate=True, 
                                output_scores=False,
                                max_length=512, 
                                max_new_tokens=512, 
                                do_sample=True, 
                                top_p=top_p, 
                                temperature=temperature, 
                                no_repeat_ngram_size=3
                                )
        # 解码
        out_text = self.ChatYuan_tokenizer.batch_decode(out[0], skip_special_tokens=True)
        # 返回结果
        return self.postprocess(out_text[0])
