# Model Evaluation

<p align="center">
<a href="https://modelscope.cn/studios/AI-ModelScope/Evaluation-Model/summary"><img src="https://img.shields.io/badge/ModelScope-blueviolet" alt="modelscope"></a>
<a href="https://aistudio.baidu.com/aistudio/projectdetail/6145966"><img src="https://img.shields.io/badge/-AIStudio-337AFF" alt="AIStudio"></a>
</p> 

<p align="center">  
 <a href="https://github.com/thomas-yanxin/LLM-EVALUATION/blob/master/docs/Evaluation_dimensions.md"><strong>评价维度</strong></a>| <a href="https://github.com/thomas-yanxin/LLM-EVALUATION/blob/master/docs/update_history.md"><strong>更新日志</strong></a>  

</p>

## 👀项目介绍

本项目基于[ModelScope(魔搭)](https://modelscope.cn/studios/AI-ModelScope/Evaluation-Model/summary)社区和[飞桨AIStudio](https://aistudio.baidu.com/aistudio/projectdetail/6145966)社区, 依托平台巨大的用户规模, 通过开源模型推理或者API接入的方式, 探索为开发者提供针对LLM的测评体验. 其能够对于某个prompt基于不同的模型生成多个结果, 开发者能基于生成结果比较模型效果.

目前仍是雏形, 还在系统规划当中, 致力为中文大模型社区提供一个**机制尽可能公开透明**、**标准尽可能全面准确**、**结果尽可能真实权威**的大模型评价机制.

## 🔥测评维度

| 模型名称 | 参数 | 研究单位 | 开源/API | 是否商用 | ModelScope | AIStudio | 效果 |
|:----:| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| [ChatYuan-large-v2](https://github.com/clue-ai/ChatYuan) | 0.7B | [元语智能](https://github.com/clue-ai) | 开源 | 否 | √ | √ |  |
| [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) | 6B | [智谱·AI](https://maas.aminer.cn/) | 开源 | 否  | √ | √ |  |
| [Vicuna-7B](https://github.com/lm-sys/FastChat) | 7B | [lm-sys](https://lmsys.org/) | 开源 | 否  | todo | × |  |
| [rwkv-4-raven](https://huggingface.co/BlinkDL/rwkv-4-raven) | 14B | [BlinkDL](https://huggingface.co/BlinkDL) | 开源 | 是  | todo | × |  |
| [MiniMax](https://api.minimax.chat/) | 未知 | [Minimax](https://api.minimax.chat/) | 否 | 否 | × | × |  |

## 💪号召

目前针对中文领域的大模型仍缺乏**结果权威的、公开透明**的评价体系. 本项目致力于针对此领域进行探索, 希望和开源社区一道, **从第三方的中立视角**, 采用**社区公认且公开透明的评测机制**, **尽可能客观、准确地**评价中文领域的大模型.

目前此项目仍是雏形, 评价维度、系统建设、人类打分、算力支持等各个方面均需要开源社区的力量一同参与. 希望感兴趣的同学能够为此项目以Issue、PR等形式提供灵感、方案、代码贡献!

## 📖参考资料

1. 聊天机器人竞技场排行榜: https://lmsys.org/blog
2. SuperCLUE: https://github.com/CLUEbenchmark/SuperCLUE
