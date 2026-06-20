# 术语表 (Glossary)

## ReAct Agent 本地推理系统

> 版本: v1.0 | 创建日期: 2026-06-20 | 最后更新: 2026-06-20
> 关联文档: [BRD](BRD.md) | [FSD](FSD.md) | [ISSUE_TRACKING](ISSUE_TRACKING.md)

---

## GGUF (GPT-Generated Unified Format)

### 定义
GGUF 是一种量化模型文件格式，由 llama.cpp 项目作者 Georgi Gerganov 设计，于 2023 年 8 月取代旧的 GGML 格式。它将几百 GB 的原始大模型压缩为可在消费级硬件上运行的单个文件。

### 核心价值
把几百 GB 的原始大模型压缩到能在消费级硬件上运行的单个文件。

```
原始模型 (FP16):  ~70GB
         ↓  量化 (Quantization)
GGUF (Q4_K_M):   ~22GB   ← 体积缩小 3x，精度损失 < 2%
```

### 量化原理
LLM 的权重参数原本用 16 位浮点数 (FP16) 存储。量化就是把每个参数映射到更少的比特位。

| 量化级别 | 每参数位数 | 35B 模型体积 | 质量损失 | 适用场景 |
|---------|-----------|-------------|---------|---------|
| FP16 (原始) | 16 bit | ~70 GB | 0% | 数据中心 GPU |
| Q8_0 | 8 bit | ~35 GB | < 0.5% | 高端工作站 |
| Q5_K_M | 5 bit | ~24 GB | < 1% | 中端设备 |
| **Q4_K_P** | 4 bit | **~22 GB** | < 2% | 边缘设备 (当前使用) |
| Q2_K | 2 bit | ~12 GB | 3-5% | 极端受限设备 |

### 文件结构
一个 `.gguf` 文件包含:

```
┌──────────────────────────┐
│  Magic Number (GGUF)      │  ← 文件格式标识
├──────────────────────────┤
│  Header (元数据)           │
│   - 模型名/架构            │
│   - 词汇表大小             │
│   - 上下文长度             │
│   - 层数/头数              │
│   - tokenizer 类型         │
│   - chat_template         │
├──────────────────────────┤
│  Tensor 数据 (量化权重)     │  ← 占文件 99%+ 体积
│   - 每层的 Q/K/V/O 矩阵   │
│   - 量化 scale/zero-point │
├──────────────────────────┤
│  Padding (对齐)            │
└──────────────────────────┘
```

### 为什么本项目使用 GGUF
1. **Jetson Orin 内存仅 64GB** — FP16 的 70GB 装不下，Q4 量化的 22GB 刚好
2. **llama-cpp-python 原生支持** — 无需 PyTorch/Transformers 等重型依赖
3. **纯 CPU/边缘 GPU 推理** — 不依赖 CUDA 虚拟化或云端 GPU
4. **单文件分发** — 一个 `.gguf` 文件包含模型权重 + tokenizer + 配置，即下即用

### 当前使用的模型文件
```
Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf  (22GB)
    ↑              ↑            ↑
 模型名         社区微调版     Q4 量化 + K-Quants + 优化排列
```

- **Qwen3.6-35B**: 阿里通义千问 3.6 版本，350 亿参数
- **A3B**: MoE (Mixture of Experts) 架构，每次推理激活 3B 参数
- **Uncensored-HauhauCS-Aggressive**: 社区无条件版本，响应风格更直接
- **Q4_K_P**: Q4 量化 + K-Quants 重要性感知量化 + 性能优化排列

---

## ReAct (Reasoning + Acting)

### 定义
ReAct 是一种将推理 (Reasoning) 和行动 (Acting) 交织的 Agent 范式。LLM 通过 Thought→Action→Action Input→Observation 的四步循环完成复杂推理任务。

### 四元组

| 步骤 | 产生者 | 含义 | 示例 |
|------|--------|------|------|
| **Thought** | LLM | 推理：分析当前状态，决定下一步 | "I need to multiply 123 by 45" |
| **Action** | LLM | 决策：选择调用哪个工具 | "calculator" |
| **Action Input** | LLM | 参数：传给工具的输入 | "123 * 45" |
| **Observation** | 工具 | 反馈：工具执行的实际结果 | "5535" |

### 循环终止条件
- **Final Answer**: LLM 判断已获得足够信息，输出最终答案
- **Max Steps**: 达到配置的最大循环步数（默认 8），安全终止

### 错误恢复
解析失败时，Agent 不终止，而是将失败的输出作为 Observation 反馈给 LLM，提示 "try again"，让 LLM 自行纠正格式。

---

## Chat Template

### 定义
Chat Template 是 LLM 用于区分对话角色的 special token 模板。它告诉模型哪些部分是系统指令，哪些是用户输入，哪些是历史对话。

### Qwen2 Chat Template

```
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is 123 × 45?
<|im_end|>
<|im_start|>assistant
I need to use the calculator.
Action: calculator
Action Input: 123 * 45
```

| Token | 含义 |
|-------|------|
| `<\|im_start\|>` | 消息开始 (IM = Instant Message) |
| `<\|im_end\|>` | 消息结束 |
| `system` | 系统指令（角色定义、格式规则） |
| `user` | 用户输入 |
| `assistant` | 模型回复 |

### 其他常见 Chat Template

| 模型系列 | 模板格式 |
|---------|---------|
| Qwen2 | `<\|im_start\|>role\n...<\|im_end\|>` |
| Llama 3 | `<\|begin_of_text\|><\|start_header_id\|>role<\|end_header_id\|>\n...<\|eot_id\|>` |
| ChatML | `<\|im_start\|>role\n...<\|im_end\|>` (与 Qwen2 相似) |

---

## KV-Cache (Key-Value Cache)

### 定义
KV-Cache 是 Transformer 推理中的键值缓存机制。当模型逐 token 生成响应时，之前所有 token 的 Key 和 Value 矩阵被缓存在内存中，避免重复计算。

### 与 Prompt 复用的关系

```
第一轮 Prompt: [System Prompt (500ch)] [User: q1 (30ch)] [History: (10ch)]
第二轮 Prompt: [System Prompt (500ch)] [User: q2 (50ch)] [History: (50ch)]
                ↑                     ↑                    ↑
              完全复用               新内容               前 10ch 复用
              (same)                (new)               后 40ch 新增
```

**Prompt 复用率** = 公共前缀长度 / 总 prompt 长度 × 100%

在本项目中，实测 Qwen3.6 35B 的 prompt 复用率约 60-90%（取决于对话轮次和 user message 的变化频率）。

---

## Flash Attention

### 定义
Flash Attention 是注意力计算的 IO-aware 优化算法，通过分块计算 (tiling) 和重计算 (recomputation) 减少 GPU 高带宽内存 (HBM) 的读写次数。

### 性能收益
- 标准 Attention: O(n²) 内存访问
- Flash Attention: O(n) 内存访问
- Jetson Orin 实测: 延迟降低 40-60%

---

## RoPE (Rotary Position Embedding)

### 定义
旋转位置编码，一种将位置信息注入 token 表示的技术。通过对 Q 和 K 向量施加旋转变换来编码 token 的相对位置。

### RoPE Scaling
当推理上下文超过训练长度时，需要对 RoPE 的频率进行缩放:

| 类型 | 值 | 说明 |
|------|-----|------|
| NONE | 0 | 无缩放（训练长度内） |
| LINEAR | 1 | 线性缩放（本项目使用） |
| YARN | 2 | YaRN 缩放（更好的长上下文支持） |

---

## BPE (Byte Pair Encoding)

### 定义
字节对编码，一种子词 (subword) 分词算法。通过统计字符对出现频率、迭代合并高频对来构建词汇表。

### 示例
```
原始文本: "agentization"
BPE 分词: ["agent", "ization"]
        ↑ 高频"agent"保留完整  ↑ 后缀"ization"也是独立 subword
```

### 与本项目的关系
Token 统计依赖 llama.cpp 内置的 BPE tokenizer。Diff 算法中的 token 估算（3.5 ch/token）是基于 BPE 中文混合文本的经验值。

---

## 其他术语

| 术语 | 定义 |
|------|------|
| **llama.cpp** | C/C++ 实现的 LLM 推理引擎，支持 GGUF 格式，可在 CPU/边缘 GPU 上运行 |
| **llama-cpp-python** | llama.cpp 的 Python 绑定，提供 `Llama()` 类 |
| **langchain** | LLM 应用开发框架，本项目仅使用其 `@tool` 装饰器 |
| **MoE** | Mixture of Experts，将模型拆分为多个专家子网络，每次推理只激活部分专家 |
| **OOM** | Out of Memory，内存不足错误 |
| **OOV** | Out of Vocabulary，词汇表外词（BPE 子词分词可缓解） |

---

> 最后更新: 2026-06-20
