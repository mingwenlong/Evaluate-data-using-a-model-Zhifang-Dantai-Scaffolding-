# Evaluate-data-using-a-model-Zhifang-Dantai-Scaffolding-
智方丹台评分脚手架，用模型评价数据
[README.md](https://github.com/user-attachments/files/23953545/README.md)
# 智方丹台评分脚手架

本项目实现使用本地 `qwen2.5-7B` 按照 Markdown 评价指标，对模型生成的 `output` 进行逐项评分并加权汇总；`input` 仅作为评审的上下文依据，不参与直接评分。

## 环境准备
- 操作系统：Windows
- 安装 Python 3.10+
- 安装依赖：
```
pip install -r requirements.txt
```

## 模型准备
- 默认使用相对路径：`models/Qwen2.5-7B-Instruct`。
- 你也可以在服务器通过环境变量 `MODEL_PATH` 指向任意模型目录（绝对或相对路径均可）。
- 在 `config/config.json` 中已配置相对路径，若不使用环境变量则按相对路径加载。

### 使用 huggingface_hub.snapshot_download 本地下载（推荐）
若在服务器上预先下载模型，请使用 `snapshot_download` 并限制必要文件，避免下载多余内容：

Python 示例：
```
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/data/models/Qwen2.5-7B-Instruct",
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "pytorch_model.bin",  # 如 safetensors 不存在，使用该文件
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "vocab.json"
    ]
)
```

下载完成后，将 `MODEL_PATH` 指向该本地目录，例如：
```
export MODEL_PATH=/data/models/Qwen2.5-7B-Instruct
```

注意：本项目在本地路径为目录时强制仅从本地加载（不访问远端）。若目录缺少关键文件，将提示缺失项并引导修复；请确保至少包含 `config.json`、`tokenizer.json/tokenizer.model/vocab.json`、以及 `model.safetensors` 或 `pytorch_model.bin`。

## 运行评估
```
python src/evaluate.py
```
- 程序将读取：
  - 评价指标：从 `docs/评估标准.md` 解析（必须存在）
- 数据集：默认 `data(longthink).json`（兼容 `data(smallthink).json`，其中 `input` 字段作为上下文，`output` 字段为被评内容）
  - 提示模板：`prompts/eval_prompt_zh.txt`
- 输出评分到 `scores.json`，包含每条样本的维度分与加权总分。

### 服务器环境变量覆盖（无需改文件）
在服务器上可以通过环境变量覆盖 `config.json` 的关键项：

- `MODEL_PATH`：模型路径（例如 `/data/models/Qwen2.5-7B-Instruct`）
- `DEVICE_MAP`：设备映射（默认 `auto`）
- `TORCH_DTYPE`：精度（推荐 `bfloat16` 或 `float16`）
- `SMALLTHINK_JSON_PATH`：数据集路径（默认 `data(longthink).json`）
- `SCORES_OUTPUT_PATH`：输出路径（默认 `scores.json`）
- `METRICS_MD_PATH`：Markdown评价标准路径（必须存在）
- `LIMIT_SAMPLES`：样本上限（如 `1` 或 `all`/`全部`）
- 其他：`MAX_NEW_TOKENS`、`MAX_INPUT_TOKENS`、`INPUT_TRUNCATE_CHARS`、`MAX_GEN_TIME_SECONDS`、`USE_CHAT_TEMPLATE`、`DO_SAMPLE`、`TEMPERATURE`、`TOP_P`、`MOCK_GENERATION`

示例（Linux bash）：
```
export MODEL_PATH=/data/models/Qwen2.5-7B-Instruct
export LIMIT_SAMPLES=1
python src/evaluate.py
```
示例（Windows PowerShell）：
```
$env:MODEL_PATH="D:/models/Qwen2.5-7B-Instruct"
$env:LIMIT_SAMPLES="1"
python src/evaluate.py
```

若你使用的是绝对路径（如 `/workspace/suanfa2/models/Qwen2.5-7B-Instruct`），无需改动代码，直接设置 `MODEL_PATH` 即可；我们已在加载逻辑中避免将本地路径误判为远程仓库 ID，从而不会出现 `HFValidationError`。

## 配置项说明（config/config.json）
- `model_path`：模型路径（默认相对 `models/Qwen2.5-7B-Instruct`，可用环境变量覆盖）。
- `device_map`：设备映射（默认 `auto`）。
- `torch_dtype`：精度（默认 `bfloat16`，在CPU上按框架默认；T4 建议 `float16`）。
- `max_new_tokens`：最大生成长度（默认 `128`；若需评估全部指标，建议 `768–1024`）。
- `temperature`：温度（默认 `0.2`，评审建议偏低）。
- `do_sample`：是否采样（默认 `false`，评审建议关闭）。
- `top_p`：采样概率阈值（默认 `0.9`，仅在采样开启时生效）。
- `use_chat_template`：是否使用聊天模板（默认 `true`）。
- `input_truncate_chars`：输入字符截断上限（默认 `2000`）。
- `max_input_tokens`：Tokenizer 输入 token 上限（默认 `1024`；若显存允许，建议 `2048`）。
- `max_gen_time_seconds`：生成超时/早停上限（默认 `20` 秒；按样本计时，建议 `150–240`，不关心上限可设更大如 `600`）。
- `metrics_md_path`：评价指标 Markdown 路径（默认 `docs/评估标准.md`，存在时优先使用）。
- `smallthink_json_path`：数据集路径（默认 `data(longthink).json`，兼容 `data(smallthink).json`）。
- `scores_output_path`：评分输出文件路径（默认 `scores.json`）。
- `limit_samples`：评估样本上限。设为整数限制数量；设为 `all` 或 `全部` 表示评估全部样本。
- `limit_metrics`：用于限制参与评估的指标数量，从解析出的顺序取前 N 项；设为 `0` 或不配置表示使用全部指标（默认 `3`，便于快速调试与管线验证）。
- `mock_generation`：模拟评估模式（默认 `false`；启用时不加载模型，直接返回示例结构，用于验证管线）。

### 参数作用范围（按样本 vs 全局）
- 按样本生效：`max_input_tokens`、`input_truncate_chars`、`max_new_tokens`、`max_gen_time_seconds`、`do_sample`、`temperature`、`top_p`、`use_chat_template`（每个样本一次生成时应用）。
- 全局生效：`model_path`、`device_map`、`torch_dtype`、`metrics_md_path`、`smallthink_json_path`、`scores_output_path`、`limit_samples`（样本范围）、`limit_metrics`（维度集合）、`mock_generation`（是否模拟）。

### 生成与解析行为（重要）
- 早停与超时：每个样本在生成时会受 `max_new_tokens` 与 `max_gen_time_seconds` 限制；达到上限会立即停止并尝试解析当前文本。
- 完整性判断：生成检测到顶层 JSON 且包含 `dimensions`，当维度数量达到提示中的预期数量时即可认为“完整”；不再强制要求模型输出 `weighted_total`。
- 总分处理：若模型未输出 `weighted_total` 或数值不一致，脚本会基于维度的 `score/max_score/weight` 自动重新计算并填充 `weighted_total`。
- 维度缺失：若未能生成到完整维度数，归一化阶段会把缺失维度补为 0 分，并在 `reason` 中标注“未评估或缺失，默认0分”。
  - 兼容性：解析器已兼容 `dimensions` 为字典且值为数值的形式（如 `{ "配伍禁忌符合率": 1.0 }`），不再因缺少嵌套对象而视为未评估。

### T4 GPU 推荐配置（评估全部指标）
- `torch_dtype`: `float16`
- `device_map`: `auto`
- `max_new_tokens`: `768`（必要时 `1024`）
- `max_gen_time_seconds`: `150–240`（不关心时长可设到 `600`）
- `do_sample`: `false`
- `temperature`: `0.0`
- `use_chat_template`: `true`
- `input_truncate_chars`: `2000–3000`
- `max_input_tokens`: `2048`
- `limit_samples`: `all`（或 `全部`）
- `limit_metrics`: `0`（全部指标）

### 全量评估示例（Windows PowerShell）
```
$env:MODEL_PATH="D:/models/Qwen2.5-7B-Instruct"
$env:DEVICE_MAP="auto"
$env:TORCH_DTYPE="float16"
$env:MAX_NEW_TOKENS="768"
$env:MAX_GEN_TIME_SECONDS="240"
$env:DO_SAMPLE="false"
$env:TEMPERATURE="0.0"
$env:USE_CHAT_TEMPLATE="true"
$env:INPUT_TRUNCATE_CHARS="3000"
$env:MAX_INPUT_TOKENS="2048"
$env:LIMIT_SAMPLES="all"
$env:LIMIT_METRICS="0"
python src/evaluate.py
```

### 排错指南（维度缺失/全为0的情况）
- 现象：`scores.json` 中 `model_raw` 为不完整 JSON 或只有部分维度闭合，出现“未评估或缺失，默认0分”。
- 处理：
  - 增大 `max_new_tokens`（如到 `1024`）与 `max_gen_time_seconds`（如到 `600`）。
  - 保持 `do_sample=false` 与 `temperature=0.0` 提升 JSON 闭合稳定性。
  - 开启 `use_chat_template=true`，减少结构不一致导致的解析失败。
  - 临时降低 `limit_metrics`（如 `3`）验证闭合与解析链路后再恢复为全部。


## 注意事项
- Markdown 指标格式要求：每个指标以 `###` 标题开头，并包含“评分范围：”与“权重：”字段，其余解释性字段将合并为描述；若缺失，将默认满分 `10`、权重 `1`。
- 若本地为CPU推理，速度较慢；建议使用具有显存的GPU并设置 `device_map=auto`。
- 仅进行“LLM 作为评审”的主观语义评分，若需加入规则校验（如模块缺失、禁忌短语），可在评审模板中进一步约束或扩展脚本。
 - 若出现 NumPy/SciPy 版本警告（如 `A NumPy version >=1.22.4 and <2.3.0 is required`），可执行以下任一操作以消除警告：
   - 本项目不依赖 `scikit-learn`/`scipy`，可卸载：`pip uninstall -y scikit-learn scipy`；或
   - 使版本匹配：`pip install "numpy<2.3"`（或升级 `scipy` 至兼容版本）。
### 相对路径默认值（已在 config/config.json 中设置）
- `model_path`: `models/Qwen2.5-7B-Instruct`
- `metrics_md_path`: `docs/评估标准.md`
- `smallthink_json_path`: `data(longthink).json`
- `scores_output_path`: `scores.json`

若你在服务器上使用不同目录结构，可以用上述环境变量进行覆盖。
