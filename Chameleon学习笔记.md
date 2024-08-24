# Chameleon 学习笔记

Chameleon: Mixed-Modal Early-Fusion Foundation Models **一切皆Token**

一句话概述：完成了一类生成模型，在训练时不去区分文本和图像的token，最终能输出文本、图像自然结合的内容

https://github.com/facebookresearch/chameleon

https://arxiv.org/abs/2405.09818

## Introduction

Chameleon: a **family **of **early-fusion** token-based mixed-modal models capable of understanding and generating images and text **in any arbitrary **.

Early Fusion: 所有模态（图像、文本、代码、视频）从一开始就被投影到一个共享的表示空间中。这种方法允许模型在不同模态之间无缝地进行推理和生成。

Token-Based: 将图像Encode为离散的标记，类似于文本中的单词，可以使用相同的Transformer架构来处理图像和文本标记序列。



（在这里先展示一个我认为非常厉害的output）

![image-20240824204009939](D:\clash\image-20240824204009939.png)



**motivation**：近些年大多多模态模型的思想停留在使用modality specific encoders or decode来对各种模态分别建模，这限制了它们整合跨模态信息的能力，难以生成包含任意顺序图像和文本的多模态文档。在该工作中，Meta团队的Chameleon提出了一个能够生成和推理任意交错文本和图像内容混合序列的混合模态基础模型家族（效果如下图），致力于实现完整的多模态文档建模（也就是把文本、图像等多模态信息在sequence中自然地融合不作任何区分）。

![image-20240824120221130](D:\clash\image-20240824120221130.png)

可以看到在这张图里，Chameleon给出了非常自然的回答，在合适的文本处插入了合适的图片。



为了实现上述效果，Chameleon is instead designed to be mixed-model from inception and uses a uniform architecture trained from scratch in an **end-to-end fashion on an interleaved mixture of all modalities, i.e., images, text, and code**. Chameleon 采用**fully token-based representation**来处理图像和文本模态（见下图）。通过将图像量化为离散令牌(类似于文本中的单词)，Chameleon 能够对图像和文本令牌序列应用相同的transformer architecture，而无需使用单独的图像/文本编码器或特定领域的解码器。这种早期融合方法，从一开始就将所有模态投射到共享的表示空间中，使得跨模态的推理和生成变得无缝。

![image-20240824125404272](D:\clash\image-20240824125404272.png)

然而，使用这样的架构也带来了额外的技术难点，尤其在optimization stability and scaling方面，method部分会介绍相关内容

实验表明，Chameleon在多种任务上表现出色：在视觉问答和图像描述基准测试中，Chameleon-34B 达到了最先进的性能，超越了Flamingo、IDEFICS和Llava-1.5等模型；在常识推理和阅读理解任务上与Mixtral 8x7B和Gemini-Pro等模型相媲美。

但最重要的是，Chameleon在混合模态推理和生成方面实现了突破。由于仅使用静态的公共基准测试来评估模型性能可能存在局限性，Meta团队额外设计了人类评估实验，通过测量混合模态长篇响应对开放式提示的质量进行评估。在成对比较中，Chameleon-34B显著优于强大的基准模型如Gemini-Pro和GPT-4V，分别在与Gemini-Pro和GPT-4V的比较中获得了60.4%和51.6%的偏好率。



## Pre-Training

### Tokenization

这部分包含两个工作：对文本的tokenization和对图像的tokenization

后者参考了Make-a-scene: Scene-based text-to-image generation with human priors, 基于VQ-VAE的思想将 512×512 的图像编码为来自大小为 8192 的codebook的 1024 个离散令牌

前者则是一个BPE tokenizer with a vocabulary size of 65,536, which includes the 8192 image codebook tokens, using the sentencepiece library.



### Pre-Training Data

作者将预训练阶段划分为两个独立的阶段：第一个阶段占训练的前80%，而第二个阶段占最后20%。

对于所有的Text-To-Image pairs，作者通过rotate处理以确保50%的情况下图像在文本之前。

#### First Stage

在该阶段，作者使用以下**超大**规模的完全无监督数据集的混合数据。

Text-Only: 多种文本数据集，包括用于训练LLaMa-2和CodeLLaMa的预训练数据，共计2.9万亿个纯文本token。

Text-Image: 图像被调整大小并中心裁剪为512×512像素进行令牌化, 总共包含14亿对文本-图像对，产生1.5万亿个文本-图像token。

Text/Image Interleaved: 从公开的网页资源中获取数据，总共获得4000亿个交错的文本和图像数据token

#### Second Stage

在这一阶段做高质量的training，将第一阶段数据的权重降低50%，并混入质量更高的数据集，并加入了大量指令调优集的过滤后的训练集子集，同时保持类似比例的图像文本token来进行训练。



### Stability

这部分内容比较复杂，大概可以概括为：

将变色龙（Chameleon）模型扩展到超过80亿参数和1万亿令牌时，通常在训练的后期才会出现明显的不稳定性。

为了解决这一问题，作者爆改了Architecture和Optimization方案，象征性地推了推式子，然后修改了架构

架构：Llama2

Norm：RMSNorm

SwiGLU激活函数+RoPE

Softmax部分，使用QK-Norm：QK-Norm 通过对注意力内的查询和关键向量应用层Norm，直接控制输入到 softmax 的Norm增长。

扩展到34B时，需要使用Swin Transformer的Normalization策略，保证限制前馈Block的增长（前馈Block可能导致SwiGLU产生问题）

优化部分，使用AdamW优化器，$\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-5}$, weight decay设置为0.1， 1.0 threshold的全局梯度裁剪，0.1的Dropout

关于硬件：Chameleon预训练是在Meta的RSC上进行的，对齐则是在其他内部研究集群上完成的，两个环境都使用NVIDIA A100 80GB GPU，具体情况如下：

![image-20240824205944669](D:\clash\image-20240824205944669.png)

