**简体中文** | [Origin English README](README_en.md)

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [one-glm](#one-glm)
- [GLM](#glm)
  - [Pretrained Models](#pretrained-models)
  - [Results](#results)
    - [SuperGLUE](#superglue)
    - [Seq2Seq](#seq2seq)
    - [Language Modeling](#language-modeling)
  - [Get Started](#get-started)
      - [Generation](#generation)
      - [Classification](#classification)
    - [Manual Installation](#manual-installation)
    - [Clone this repo](#clone-this-repo)
    - [Model Parallelism](#model-parallelism)
  - [Usage](#usage)
    - [Left-to-Right Generation / Blank Filling (Interactive)](#left-to-right-generation--blank-filling-interactive)
      - [Usage of `[MASK]` (Entity Prediction):](#usage-of-mask-entity-prediction)
        - [Example1](#example1)
        - [Example2 (Chinese)](#example2-chinese)
      - [Usage of `[sMASK]` (Sentence Prediction)](#usage-of-smask-sentence-prediction)
        - [Example3](#example3)
        - [Example4 (Chinese)](#example4-chinese)
      - [Usage of `[gMASK]` (Long Text Generation)](#usage-of-gmask-long-text-generation)
        - [Example5 (Chinese)](#example5-chinese)
        - [Example1](#example1-1)
        - [Example2 (Chinese)](#example2-chinese-1)
    - [SuperGLUE](#superglue-1)
    - [Seq2Seq](#seq2seq-1)
      - [Train with your own data](#train-with-your-own-data)
    - [Multiple Choice (Zero-shot)](#multiple-choice-zero-shot)
    - [Language Modeling](#language-modeling-1)
      - [LAMBADA Cloze Accuracy](#lambada-cloze-accuracy)
      - [LM Perplexity](#lm-perplexity)
    - [Text Infilling](#text-infilling)
  - [Pretrain](#pretrain)
  - [Citation](#citation)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


# one-glm
将 https://github.com/THUDM/GLM 改成 OneFlow 后端运行， 获得大幅度的训练速度提升。

下面分别是在 A100 PCIE 40G 和 GTX 3090 上 one-glm 相比于原始的 THUDM/GLM 的预训练模型的性能表现：

![图片](https://user-images.githubusercontent.com/35585791/209784371-b6257ac0-83dd-4a23-b5ce-05508bdda29c.png)

![图片](https://user-images.githubusercontent.com/35585791/209784588-0e72d49e-c5c3-4b4b-94ad-569f407d9716.png)

![图片](https://user-images.githubusercontent.com/35585791/209786126-54d19d39-ef07-47b6-bf29-1b5bb6a4eb07.png)

![图片](https://user-images.githubusercontent.com/35585791/209786155-445a3eff-a01b-4652-9e6b-48a11b3555ff.png)

# GLM 


GLM是一种通用语言模型，使用自回归填空目标进行预训练，被使用在各种自然语言理解和生成任务中。

请参考我们的论文了解GLM的详细描述：
[GLM: General Language Model Pretraining with Autoregressive Blank Infilling](https://arxiv.org/abs/2103.10360) (ACL
2022)

Zhengxiao Du*，Yujie Qian*，Xiao Liu，Ming Ding，Jiezhong Qiu，Zhilin Yang，Jie Tang（*: 相等贡献）

我们发布了[GLM-130B](https://github.com/THUDM/GLM-130B) ，这是一个基于GLM框架的双语（英文和中文）预训练语言模型，具有130亿个参数。

## Pretrained Models
> 预训练模型

您可以从[OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/duzx16_mails_tsinghua_edu_cn/Eg8MZe62MlVFs_mK2tHaH-sBC-UC01jpGPZop08pID7sOw?e=MsevNR) 或 [清华云](https://cloud.tsinghua.edu.cn/d/13f5b03da9594e5490c4)下载论文中使用的预训练模型。

| Name              | Params | Language | Corpus                                                                              | Objective      | File                                                               | Config                            |
|-------------------|--------|----------|-------------------------------------------------------------------------------------|----------------|--------------------------------------------------------------------|-----------------------------------|
| GLM-Base          | 110M   | English  | Wiki+Book                                                                           | Token          | glm-base-blank.tar.bz2                                             | model_blocklm_base.sh             |
| GLM-Large         | 335M   | English  | Wiki+Book                                                                           | Token          | glm-large-blank.tar.bz2                                            | model_blocklm_large.sh            |
| GLM-Large-Chinese | 335M   | Chinese  | Wiki+Book                                                                           | Token+Sent+Doc | glm-large-chinese.tar.bz2                                          | model_blocklm_large_chinese.sh    |
| GLM-Doc           | 335M   | English  | Wiki+Book                                                                           | Token+Doc      | glm-large-generation.tar.bz2                                       | model_blocklm_large_generation.sh |
| GLM-410M          | 410M   | English  | Wiki+Book                                                                           | Token+Doc      | glm-1.25-generation.tar.bz2                                        | model_blocklm_1.25_generation.sh  |
| GLM-515M          | 515M   | English  | Wiki+Book                                                                           | Token+Doc      | glm-1.5-generation.tar.bz2                                         | model_blocklm_1.5_generation.sh   |
| GLM-RoBERTa       | 335M   | English  | RoBERTa                                                                             | Token          | glm-roberta-large-blank.tar.bz2                                    | model_blocklm_roberta_large.sh    |
| GLM-2B            | 2B     | English  | [Pile](https://arxiv.org/abs/2101.00027)                                            | Token+Sent+Doc | glm-2b.tar.bz2                                                     | model_blocklm_2B.sh               |
| GLM-10B           | 10B    | English  | [Pile](https://arxiv.org/abs/2101.00027)                                            | Token+Sent+Doc | [Download](https://lfs.aminer.cn/misc/cogview/glm-10b-1024.zip)    | model_blocklm_10B.sh              |
| GLM-10B-Chinese   | 10B    | Chinese  | [WuDaoCorpora](https://www.sciencedirect.com/science/article/pii/S2666651021000152) | Token+Sent+Doc | [Download](https://lfs.aminer.cn/misc/cogview/glm-10b-chinese.zip) | model_blocklm_10B_chinese.sh      |


将下载的文件解压缩到本地文件夹中，并将相应脚本中的 `CHECKPOINT_PATH` 设置为文件夹路径。

## Results
> 结果

### [SuperGLUE](https://super.gluebenchmark.com)

开发集，单模型，单任务微调(dev set, single model, single-task finetuning)

| Model                                                                                        | COPA | WSC  | RTE  | WiC  | CB        | MultiRC   | BoolQ | ReCoRD    |
|----------------------------------------------------------------------------------------------|------|------|------|------|-----------|-----------|-------|-----------|
| GLM-10B                                                                                      | 98.0 | 95.2 | 93.1 | 75.7 | 98.7/98.2 | 88.1/63.3 | 88.7  | 94.4/94.0 |
| [DeBERTa-XXLarge-v2](https://github.com/microsoft/DeBERTa/tree/master/experiments/superglue) | 97.0 | -    | 93.5 | -    | -         | 87.8/63.6 | 88.3  | 94.1/93.7 |

### Seq2Seq

> Seq2Seq模型是输出的长度不确定时采用的模型，这种情况一般是在机器翻译的任务中出现，将一句中文翻译成英文，那么这句英文的长度有可能会比中文短，也有可能会比中文长，所以输出的长度就不确定了。来源： https://zhuanlan.zhihu.com/p/194308943


[CNN/Daily Mail](https://github.com/abisee/cnn-dailymail) (test set, no additional data used)

| Model         | ROUGE-1  | ROUGE-2  | ROUGE-L  |
|---------------|----------|----------|----------|
| GLM-10B       | **44.7** | 21.4     | **41.4** |
| T5-11B        | 43.5     | **21.6** | 40.7     |
| PEGASUS-Large | 44.2     | 21.5     | **41.4** |
| BART-Large    | 44.2     | 21.3     | 40.9     |

[XSum](https://github.com/EdinburghNLP/XSum) (test set, no additional data used)

| Model         | ROUGE-1  | ROUGE-2  | ROUGE-L  |
|---------------|----------|----------|----------|
| GLM-10B       | **48.9** | **25.7** | **40.4** |
| PEGASUS-Large | 47.2     | 24.6     | 39.3     |
| BART-Large    | 45.1     | 22.3     | 37.3     |

### Language Modeling

test set, zero-shot

| Model              | LAMBADA (accuracy) | Wikitext103 (perplexity) |
|--------------------|--------------------|--------------------------|
| GLM-10B (bi)       | 72.35              | 11.33                    |
| GLM-10B (uni)      | 67.18              | 12.22                    |
| GPT-2              | 52.66              | 17.48                    |
| Megatron-LM (8.3B) | 66.51              | 10.81                    |
| Turing-NLG         | 67.98              | 10.21                    |


## Get Started
#### Generation
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
model = model.half().cuda()
model.eval()

# Inference
inputs = tokenizer("Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.", return_tensors="pt")
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
inputs = inputs.to('cuda')
outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
print(tokenizer.decode(outputs[0].tolist()))

# Training
inputs = tokenizer(
    ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"],
    return_tensors="pt", padding=True)
inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8)
inputs = inputs.to('cuda')
outputs = model(**inputs)
loss = outputs.loss
logits = outputs.logits
```
#### Classification
```python
from transformers import AutoTokenizer, AutoModelForMultipleChoice
tokenizer = AutoTokenizer.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
model = AutoModelForMultipleChoice.from_pretrained("BAAI/glm-10b", trust_remote_code=True)
model = model.half().cuda()
model.eval()

inputs = tokenizer(["Tsinghua University is located in [MASK].",
                    "One minus one equals zero, is it correct? Answer: [MASK]"], return_tensors="pt", padding=True)
choices = [["Beijing", "Shanghai"], ["Yes", "No"]]
inputs = tokenizer.build_inputs_for_multiple_choice(inputs, choices)
inputs = inputs.to('cuda')
outputs = model(**inputs)
logits = outputs.logits
```

### Manual Installation
> 手动安装

请先安装PyTorch（我们使用的是1.7.0版本）和 [apex](https://github.com/NVIDIA/apex)，然后通过 `pip install -r requirements.txt`安装其他依赖项。


### Clone this repo
>  克隆这个仓库

```shell
git clone https://github.com/THUDM/GLM
cd GLM
```


### Model Parallelism

> 模型并行

如果您遇到CUDA out of memory错误，这意味着您的GPU内存有限，您可以尝试模型并行来将参数分给多个GPU。

以双向模型并行为例。首先运行 `change_mp.py` 来分割 checkpoint：

```shell
python change_mp.py path_to_the_checkpoint 2
```
然后在模型配置文件中更新 `checkpoint` 路径（例如 [config_tasks/model_blocklm_10B.sh](config_tasks/model_blocklm_10B.sh))，
并在脚本中将 `MP_SIZE` 更改为2（例如[scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh))。



## Usage
> 使用指南

我们提供了一些用于对GLM进行微调的脚本，以便在一些下游任务上使用。

### Left-to-Right Generation / Blank Filling (Interactive)
> 从左到右生成/填空（交互式）

将 `CHECKPOINT_PATH` 更改为您的本地路径。运行以下脚本

```
bash scripts/generate_block.sh config_tasks/model_blocklm_10B_chinese.sh
```

有些模型（GLM-2B，GLM-10B和GLM-10B-Chinese）使用了三种不同的掩码标记：“[MASK]”用于短的填空，“[sMASK]”用于句子填空，“[gMASK]”用于从左到右的生成。



<details>
<summary><b>Examples</b></summary>

#### Usage of `[MASK]` (Entity Prediction):

##### Example1

Context: Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or
SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.

GLM: the stanford university

##### Example2 (Chinese)

Context: 凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。

GLM:拿破仑军队攻克米兰城

#### Usage of `[sMASK]` (Sentence Prediction)

##### Example3

Context: There have been various types of pretraining architectures including autoencoding models (e.g., BERT),
autoregressive models (e.g., GPT), and encoder-decoder models (e.g., T5). [sMASK] We propose a General Language Model (
GLM) based on autoregressive blank infilling to address this challenge. GLM improves blank filling pretraining by adding
2D positional encodings and allowing an arbitrary order to predict spans, which results in performance gains over BERT
and T5 on NLU tasks. Meanwhile, GLM can be pretrained for different types of tasks by varying the number and lengths of
blanks. On a wide range of tasks across NLU, conditional and unconditional generation, GLM outperforms BERT, T5, and GPT
given the same model sizes and data, and achieves the best performance from a single pretrained model with 1.25×
parameters of BERT Large , demonstrating its generalizability to different downstream tasks.

GLM: However, there is a growing need to develop a single pretraining model that is not only good at natural language
understanding (NLU) or dialog generation/generation (dialog), but is also able to predict other tasks such as sentiment
analysis, conditional generation, or machine translation (MT).

##### Example4 (Chinese)

Context: 工业互联网（Industrial
Internet）是新一代信息通信技术与工业经济深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK]
它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成平台化设计、智能化制造、网络化协同、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。

GLM: 工业互联网是制造业技术、管理、模式的重大变革,是推动互联网、大数据、人工智能和实体经济深度融合的重要载体,是建设制造强国和网络强国的重要基础。

#### Usage of `[gMASK]` (Long Text Generation)

##### Example5 (Chinese)

Context: 问题：冬天，中国哪座城市最适合避寒？问题描述：能推荐一些国内适合冬天避寒的城市吗？回答用户：旅游爱好者 回答： [gMASK]

GLM:
海南三亚,三亚的冬天是三亚的旺季,因为冬天国内大部分地方都有雾霾,而三亚却是没有雾霾的地方,所以三亚是冬天最惬意不过的地方了。在东北长大的我觉得三亚简直就是一个天堂,冬天去海口三亚旅游,享受冬天,享受阳光沙滩。但是海口却是很干燥,需要多喝水。
三亚冬天最热门的玩法就是晒太阳,在海边晒晒太阳,在沙滩上晒晒太阳,感觉整个人都得到了解放。三亚还有一个特色项目,就是海上冲浪,在三亚的沙滩上冲浪也是一件非常刺激的事情。
海口,海口冬季的阳光十分温暖,海南的冬季也是属于冬季旅游的旺季。冬季的海口最棒的是去海南的热带野生动植物园,那里有数之不尽的热带小动物,在这里可以近距离的和它们接触,海南的热带野生动植物园也是海南的天然氧吧。还可以在海口观澜湖公园里感受海口美丽的海景。
贵阳,贵州的冬天也是十分温暖的,贵阳也是冬季避寒很好的城市之一。冬季去贵阳玩一定要去黔灵山,黔灵山是贵州香火很旺盛的一个寺庙,寺庙的冬季香火鼎盛,在冬季去寺庙游玩也是一个很好的体验。除了黔灵山,贵阳在冬季还有花溪公园可以去玩,花溪公园也是去当地公园玩最好的选择。
青岛,青岛的冬天是青岛最舒服的时候,青岛有很多海滨浴场,冬天去海边泡一泡温泉,然后晒晒太阳是一件十分惬意的事情。青岛也有沙滩,冬天在沙滩上晒晒太阳,看看海,再玩玩沙滩游戏,感觉十分快乐的事。
</details>



您也可以在单个示例中添加多个 `[MASK]` 和 `[sMASK]`。 模型将从左到右依次填充空白。 每个空白的答案总是以一个特殊字符串开头。


<details>
<summary><b>Examples</b></summary>

##### Example1

Context: There have been various types of pretraining architectures including autoencoding models (e.g., BERT), autoregressive models (e.g., GPT), and [MASK] (e.g., T5). [sMASK] We propose a General Language Model ( GLM) based on autoregressive blank infilling to address this challenge. GLM improves blank filling pretraining by adding 2D positional encodings and allowing an arbitrary order to predict spans, which results in performance gains over [MASK] on NLU tasks. Meanwhile, GLM can be pretrained for different types of tasks by varying the number and lengths of blanks. On a wide range of tasks across NLU, conditional and [MASK], GLM outperforms BERT, T5, and GPT given the same model sizes and data, and achieves the best performance from a single pretrained model with 1.25× parameters of BERT Large , demonstrating its generalizability to different downstream tasks.

GLM: <|startofpiece|> blank filling models<|startofpiece|> However, most of them cannot easily transfer to other downstream tasks due to the different characteristics of these tasks.<|startofpiece|> other pretrained models<|startofpiece|> unconditional reading, and semantic role labeling tasks

##### Example2 (Chinese)

Context: 工业互联网（Industrial Internet）是新一代[MASK]与[MASK]深度融合的新型基础设施、应用模式和工业生态，通过对人、机、物、系统等的全面连接，构建起覆盖全产业链、全价值链的全新制造和服务体系，为工业乃至产业数字化、网络化、智能化发展提供了实现途径，是第四次工业革命的重要基石。[sMASK] 它以网络为基础、平台为中枢、数据为要素、安全为保障，既是工业数字化、网络化、智能化转型的基础设施，也是互联网、大数据、人工智能与实体经济深度融合的应用模式，同时也是一种新业态、新产业，将重塑企业形态、供应链和产业链。当前，工业互联网融合应用向国民经济重点行业广泛拓展，形成[MASK]、智能化制造、[MASK]、个性化定制、服务化延伸、数字化管理六大新模式，赋能、赋智、赋值作用不断显现，有力的促进了实体经济提质、增效、降本、绿色、安全发展。

GLM:
<|startofpiece|>信息技术(ICT)<|startofpiece|>工业经济(II2O)<|startofpiece|>我国工业互联网是面向工业全领域、全流程、全体系的互联网,具有多产业、多领域融合的特点。<|startofpiece|>网络化协同<|startofpiece|>平台企业

</details>

### SuperGLUE
> SuperGLUE是一个用于自然语言理解的评估基准测试，它测试语言模型的能力。 SuperGLUE包括了许多自然语言理解任务，如文本推断、语义相似性、知识图谱和翻译。它是由OpenAI等研究人员开发的。

- Translation: 下载 [SuperGlue](https://super.gluebenchmark.com/tasks) 数据并检查实验配置在 [scripts/ds_finetune_superglue.sh](scripts/ds_finetune_superglue.sh) 。请注意，需要将 `DATA_ROOT` 、`CHECKPOINT_PATH`、`SAVE_PATH`更改为本地路径。您也可以根据可用的硬件修改 `batch-size` 和 `nproc_per_node`。

- 运行以下脚本（以 COPA 数据集为例）。

```
bash scripts/ds_finetune_superglue.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```

- 我们在代码中也实现了 [P-Tuning](https://arxiv.org/abs/2103.10385)   。运行以下脚本来集成 p-tuning：

```shell
bash scripts/ds_finetune_superglue_prompt.sh \
     config_tasks/model_blocklm_10B.sh \
     config_tasks/task_copa.sh
```

- 要将 GLM 应用于具有填空微调的新 NLU 数据集，请在 [tasks/superglue/dataset.py](tasks/superglue/dataset.py)  中实现一个 `DataProcessor` ，用于数据加载，并在   [tasks/superglue/pvp.py](tasks/superglue/pvp.py) 中添加一个 `PVP` ，用于填空问题。更多细节可以在这里找到   [here](tasks/superglue/README.md) 。


### Seq2Seq

- 下载[Gigaword](https://github.com/harvardnlp/sent-summary)
  , [CNN/Daily Mail](https://github.com/artmatsak/cnn-dailymail)
  or [XSum](https://github.com/EdinburghNLP/XSum/tree/master/XSum-Dataset)  数据集，并检查实验设置在  [scripts/ds_finetune_seq2seq.sh](scripts/ds_finetune_seq2seq.sh) 中。将 `DATA_ROOT`、`CHECKPOINT_PATH`、`SAVE_PATH` 更改为本地路径

- 行以下脚本（以 CNN/Daily Mail 数据集为例）。

  ```
  bash scripts/ds_finetune_seq2seq.sh \ 
     config_tasks/model_blocklm_10B.sh \ 
     config_tasks/seq_cnndm_org.sh
  ```

- summaries 要被写入  `./runs/experiment_name/test.jsonl.hyps` 。参考文献在同一目录的 `test.jsonl.refs` 中写入。要计算 `rouge` ，请安装 [file2rouge](https://github.com/pltrdy/files2rouge)   并从 [here](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip)  下载 Stanford CoreNLP。运行以下脚本:

```
bash scripts/evaluate_seq2seq.sh \
./runs/experiment_name/test.jsonl.hyps ./runs/experiment_name/test.jsonl.refs
```

#### Train with your own data
> 用自己的数据进行训练

请将您的 `seq2seq` 数据处理成 `{split}.source` 和 `{split}.target`，每行代表一个样本的上下文 或 目标， `split` 可以是 `train` 、`val` 或 `test`。

这句话的意思是，需要将 seq2seq 的数据分为 train、val 和 test 三个部分，每个部分都包含两个文件，分别是 {split}.source 和 {split}.target。其中，{split}.source 文件包含每个样本的上下文，{split}.target 文件包含每个样本的目标。
运行下面的脚本: 

```shell
bash scripts/ds_finetune_seq2seq.sh \ 
   config_tasks/model_blocklm_10B.sh \ 
   config_tasks/seq_customization.sh
```

你可以在 `config_tasks/seq_customization.sh` 和 `config_tasks/config_blocklm_10B_cnndm.json` 中指定超参数。

### Multiple Choice (Zero-shot)
> 多项选择（零样本）


```shell
bash scripts/evaluate_multichoice.sh config_tasks/model_blocklm_10B.sh
```

注意，`CHECKPOINT_PATH` 和 `DATA_PATH` 需要更改为您的本地路径。


数据文件的每一行的格式示例如下：

```shell
{"inputs_pretokenized": "Context and question here", "choices_pretokenized": ["Choice 1", "Choice 2", "Choice 3"], "label": int}
```


### Language Modeling
> 语言建模

#### LAMBADA Cloze Accuracy
> LAMBADA 完形填空精度

* 下载[LAMBADA](https://github.com/cybertronai/bflm/blob/master/lambada_test.jsonl) 数据，并在[scripts/evaluate_lm.sh](scripts/evaluate_lm.sh) 中更改  `DATA_ROOT, CHECKPOINT_PATH`。

* 运行以下脚本:

```shell
bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_lambada.sh 
```



#### LM Perplexity
> 语言模型的困惑度

* Download
  our [test set of wikibook](https://mailstsinghuaeducn-my.sharepoint.com/:t:/g/personal/duzx16_mails_tsinghua_edu_cn/EQa_B6KY_q1FjtUeG-T52iMBFtNrfhfHcZbzMxfkJKXKRQ?e=inTdHh)
  or [Wikitext103](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) dataset and
  change `DATA_ROOT, CHECKPOINT_PATH`
  in [scripts/evaluate_lm.sh](scripts/evaluate_lm.sh)
* Run the following script
  ```shell
  bash scripts/evaluate_lm.sh \ 
     config_tasks/model_blocklm_large_generation.sh \
     config_tasks/zero_wikitext.sh 
  ```

### Text Infilling

- Download the [Yahoo](https://github.com/Varal7/blank_language_model) dataset and check the experiment setup in
  [scripts/finetune_blank.sh](scripts/finetune_blank.sh). Change `DATA_ROOT, CHECKPOINT_PATH, SAVE_PATH` to your
  local path.

- Run the following script

```
bash scripts/finetune_blank.sh \ 
     config_tasks/model_blocklm_large.sh \ 
     config_tasks/seq_blank.sh
```

## Pretrain

运行以下脚本对于使用预训练 GLM-Large 模型。

```shell
bash scripts/ds_pretrain_nvidia.sh  \ 
config/ds_block_large.sh
```

通过脚本 [scripts/ds_pretrain_nvidia.sh](scripts/ds_pretrain_nvidia.sh)  使用 DeepSpeed 启动训练程序。

您应该将 `NUM_WORKERS`  和 `NUM_GPUS_PER_WORKER` 更改，对应 `worker 的数量` 和 `每个 worker 的 GPU 数量`。

也可将 `HOST_FILE_PATH` 更改为 `OpenMPI` 样式的 hostfile 的路径。有关 DeepSpeed launcher 的更多细节，请参阅这里 [here](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node)。

文件 [config/ds_block_large.sh](config/ds_block_large.sh) 定义了预训练的超参数。 

大多数参数都很容易理解。具体举例来说，

`--train-data` 是[data_utils/corpora.py](data_utils/corpora.py) 中的 字典对象：`NAMED_CORPORA` 里面定义的多个关键字的数据集。优化器的超参数在 `config` 中的相应 json 文件中定义。json 文件的语义可以在这里找到  [here](https://www.deepspeed.ai/docs/config-json) 。



## Citation

Part of the code is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
and [PET](https://github.com/timoschick/pet).

Please cite our paper if you find this code useful for your research:

```
@article{DBLP:conf/acl/DuQLDQY022,
  author    = {Zhengxiao Du and
               Yujie Qian and
               Xiao Liu and
               Ming Ding and
               Jiezhong Qiu and
               Zhilin Yang and
               Jie Tang},
  title     = {{GLM:} General Language Model Pretraining with Autoregressive Blank Infilling},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational
               Linguistics (Volume 1: Long Papers), {ACL} 2022, Dublin, Ireland,
               May 22-27, 2022},
  pages     = {320--335},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
}
```
