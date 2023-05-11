import paddle
from paddle.distributed import fleet
from paddlenlp.transformers import (ChatGLMConfig,
                                    ChatGLMForConditionalGeneration,
                                    ChatGLMTokenizer)

batch_size = 1

def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


class ChatGLM_Predictor():
    model_name_or_path = "THUDM/chatglm-6b"
    batch_size = 1
    src_length = 128
    tgt_length = 128
    def __init__(self):
        self.tokenizer = ChatGLMTokenizer.from_pretrained(self.model_name_or_path)
        self.batch_size = self.batch_size

        tensor_parallel_degree = paddle.distributed.get_world_size()
        tensor_parallel_rank = 0
        if tensor_parallel_degree > 1:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()
            tensor_parallel_rank = hcg.get_model_parallel_rank()

        config = ChatGLMConfig.from_pretrained(self.model_name_or_path)
        paddle.set_default_dtype(config.paddle_dtype)

        self.model = ChatGLMForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            tensor_parallel_degree=tensor_parallel_degree,
            tensor_parallel_rank=tensor_parallel_rank,
            load_state_as_np=True,
            dtype=config.paddle_dtype,
        )
        self.model.eval()

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            max_length=self.src_length,
            truncation=True,
            truncation_side="left",
        )
        inputs_tensor = {}
        for key in inputs:
            inputs_tensor[key] = paddle.to_tensor(inputs[key])
        return inputs_tensor

    def infer(self, inputs):
        result = self.model.generate(
            **inputs,
            decode_strategy="sampling",
            top_k=1,
            max_length=self.tgt_length,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.end_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
        )
        result = result[0]
        return result

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            res = res.strip("\n")
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":

    predictor = ChatGLM_Predictor()
    all_texts = [
        "你好",
]
    batch_texts = batchfy_text(all_texts, batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("{}\n{}".format(text, result))

