import json
import os
import re
import difflib
import pandas as pd
from typing import List, Dict, Any

# 支持作为包模块运行（python -m src.evaluate）以及脚本直接运行（python src/evaluate.py）
try:
    from .llm_client import LocalLLMClient  # 相对导入：模块方式
except ImportError:
    from llm_client import LocalLLMClient  # 退回绝对导入：脚本方式


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Allow overriding config via environment variables on server without editing files."""
    env_map = {
        "MODEL_PATH": ("model_path", str),
        "DEVICE_MAP": ("device_map", str),
        "TORCH_DTYPE": ("torch_dtype", str),
        "METRICS_MD_PATH": ("metrics_md_path", str),
        "SMALLTHINK_JSON_PATH": ("smallthink_json_path", str),
        "SCORES_OUTPUT_PATH": ("scores_output_path", str),
        "LIMIT_SAMPLES": ("limit_samples", str),  # 支持 "all/全部" 或整数
        "LIMIT_METRICS": ("limit_metrics", int),
        "MAX_NEW_TOKENS": ("max_new_tokens", int),
        "MAX_INPUT_TOKENS": ("max_input_tokens", int),
        "INPUT_TRUNCATE_CHARS": ("input_truncate_chars", int),
        "MAX_GEN_TIME_SECONDS": ("max_gen_time_seconds", float),
        "USE_CHAT_TEMPLATE": ("use_chat_template", bool),
        "DO_SAMPLE": ("do_sample", bool),
        "TEMPERATURE": ("temperature", float),
        "TOP_P": ("top_p", float),
        "MOCK_GENERATION": ("mock_generation", bool),
    }

    def parse_bool(val: str) -> bool:
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    for env_key, (cfg_key, typ) in env_map.items():
        if env_key in os.environ and os.environ[env_key] != "":
            raw = os.environ[env_key]
            try:
                if typ is bool:
                    cfg[cfg_key] = parse_bool(raw)
                elif typ is int:
                    cfg[cfg_key] = int(raw)
                elif typ is float:
                    cfg[cfg_key] = float(raw)
                else:
                    cfg[cfg_key] = str(raw)
            except Exception:
                cfg[cfg_key] = raw
    return cfg


def read_metrics(excel_path: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(excel_path, sheet_name=0)
    cols = {c: str(c).strip() for c in df.columns}

    def find_col(candidates):
        for k in cols:
            for p in candidates:
                if p in cols[k]:
                    return k
        return None

    metric_col = find_col(["评分项目", "维度", "指标", "项目", "dimension", "metric"])
    weight_col = find_col(["权重", "weight"])
    max_col = find_col(["满分", "满分值", "评分范围", "score range", "max", "max_score"])
    # 描述列：含义、示例解释、权重设定解释，以及原来的说明/规则
    desc_cols = [
        find_col(["含义", "meaning", "definition"]),
        find_col(["示例解释", "example", "explain"]),
        find_col(["权重设定解释", "weight explain"]),
        find_col(["说明", "规则", "描述", "description", "rule"]),
    ]
    desc_cols = [c for c in desc_cols if c is not None]

    def parse_max(val) -> float:
        if val is None:
            return 10.0
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val)
        # 提取区间上界，如 0-100 / 0~1 / 0—1 / 0 到 100
        import re
        m = re.findall(r"(\d+\.?\d*)", s)
        if len(m) >= 2:
            return float(m[-1])
        if len(m) == 1:
            return float(m[0])
        # 若包含百分字样，默认 100
        if "百分" in s or "%" in s:
            return 100.0
        return 10.0

    metrics = []
    for _, row in df.iterrows():
        # 名称
        name = str(row[metric_col]).strip() if metric_col else str(row.iloc[0]).strip()
        # 跳过空名称或明显为分组标题但缺少权重/范围的行
        raw_weight = row[weight_col] if weight_col in df.columns else None
        raw_max = row[max_col] if max_col in df.columns else None
        if (name == "" or (raw_weight is None and raw_max is None)):
            continue
        # 解析权重
        try:
            weight = float(raw_weight) if raw_weight is not None and str(raw_weight) != "" else 1.0
        except Exception:
            # 文本权重（如“总权重100”），尝试提取数字
            import re
            mm = re.findall(r"(\d+\.?\d*)", str(raw_weight))
            weight = float(mm[0]) if mm else 1.0
        # 解析满分/范围
        max_score = parse_max(raw_max)
        # 描述合并并截断
        desc_parts = []
        for c in desc_cols:
            try:
                val = row[c]
                if val is not None and str(val).strip() != "":
                    desc_parts.append(str(val).strip())
            except Exception:
                pass
        desc = "；".join(desc_parts)
        short_desc = (desc[:300] + "...") if len(desc) > 300 else desc
        metrics.append({
            "name": name,
            "weight": weight,
            "max_score": max_score,
            "description": short_desc,
        })
    return metrics


def read_metrics_md(md_path: str) -> List[Dict[str, Any]]:
    """将 Markdown 版的评价标准解析为 metrics 列表。
    解析规则：
    - 每个 "### " 标题视为一个指标项，去除前缀序号，如 "1)"。
    - 在该项下，解析 "评分范围：" 获取区间上界为 max_score；
      支持格式如 `0-100`、`0–100`、`0~1`、`0 到 100`、包含百分号时取 100。
    - 解析 "权重：" 获取数字权重（支持反引号包裹和中文数字混杂，取首个数字）。
    - 将 "含义"、"示例解释"、"权重设定解释"、"说明/规则/描述" 拼接为 description（截断至300字）。
    若某项缺失，则使用默认值：max_score=10，weight=1。
    """
    if not os.path.exists(md_path):
        return []
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines()
    items = []
    current = None

    def flush_current():
        if current is None:
            return
        # 默认值回填
        name = current.get("name", "").strip()
        if not name:
            return
        max_score = current.get("max_score")
        weight = current.get("weight")
        desc = current.get("_desc_parts", [])
        meaning = str(current.get("meaning", "")).strip()
        example = str(current.get("example", "")).strip()
        weight_explain = str(current.get("weight_explain", "")).strip()
        if max_score is None:
            max_score = 10.0
        if weight is None:
            weight = 1.0
        description = "；".join([d for d in desc if d])
        short_desc = (description[:300] + "...") if len(description) > 300 else description
        items.append({
            "name": name,
            "weight": float(weight),
            "max_score": float(max_score),
            # 语义解释字段用于指导模型理解并评分
            "meaning": meaning,
            "example": example,
            "weight_explain": weight_explain,
            "description": short_desc,
        })

    def parse_max_from_text(s: str) -> float:
        if not s:
            return 10.0
        s2 = s.replace("—", "-").replace("–", "-").replace("~", "-")
        # 抽取所有数字
        nums = re.findall(r"(\d+\.?\d*)", s2)
        if len(nums) >= 2:
            try:
                return float(nums[-1])
            except Exception:
                pass
        if len(nums) == 1:
            try:
                return float(nums[0])
            except Exception:
                pass
        if ("百分" in s) or ("%" in s):
            return 100.0
        return 10.0

    def parse_weight_from_text(s: str) -> float:
        if not s:
            return 1.0
        # 先找反引号中的数字
        m = re.search(r"`\s*(\d+\.?\d*)\s*`", s)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
        # 其次找首个数字
        m2 = re.search(r"(\d+\.?\d*)", s)
        if m2:
            try:
                return float(m2.group(1))
            except Exception:
                pass
        return 1.0

    for raw in lines:
        line = raw.strip()
        # 新指标项
        if line.startswith("### "):
            flush_current()
            title = line[4:].strip()
            # 去掉前缀序号 "1) " 等
            title = re.sub(r"^\s*\d+\)\s*", "", title)
            current = {"name": title, "_desc_parts": []}
            continue
        if current is None:
            continue
        # 评分范围
        if line.startswith("- 评分范围：") or line.startswith("评分范围："):
            val = line.split("：", 1)[-1].strip()
            current["max_score"] = parse_max_from_text(val)
            continue
        # 权重
        if line.startswith("- 权重：") or line.startswith("权重："):
            val = line.split("：", 1)[-1].strip()
            current["weight"] = parse_weight_from_text(val)
            continue
        # 语义解释字段
        if line.startswith("- 含义：") or line.startswith("含义："):
            val = line.split("：", 1)[-1].strip()
            if val:
                current["meaning"] = val
            continue
        if line.startswith("- 示例解释：") or line.startswith("示例解释："):
            val = line.split("：", 1)[-1].strip()
            if val:
                current["example"] = val
            continue
        if line.startswith("- 权重设定解释：") or line.startswith("权重设定解释："):
            val = line.split("：", 1)[-1].strip()
            if val:
                current["weight_explain"] = val
            continue
        # 其他描述性字段聚合
        if any(key in line for key in ["说明：", "规则：", "描述："]):
            val = line.split("：", 1)[-1].strip()
            if val:
                current["_desc_parts"].append(val)
            continue
    # 收尾
    flush_current()
    return items


def read_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    return data


def build_prompt(template_path: str, metrics: List[Dict[str, Any]], sample: Dict[str, Any], max_chars: int) -> str:
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    # 向模型提供必要三列以及语义解释字段，帮助模型根据含义/示例/权重设定解释进行评分
    compact_metrics = []
    for m in metrics:
        item = {
            "name": str(m.get("name", "")),
            "weight": float(m.get("weight", 1.0)),
            "max_score": float(m.get("max_score", 10.0)),
        }
        # 仅传递精简版描述，避免提示过长导致指令尾部被截断
        if m.get("description"):
            item["description"] = str(m.get("description"))
        compact_metrics.append(item)
    metrics_json = json.dumps(compact_metrics, ensure_ascii=False)
    # 仅对 output 评分：不传入 input 相关内容
    output_text = str(sample.get("output") or sample.get("response") or sample.get("content") or "")
    # 截断过长输入
    output_text = output_text[:max_chars]
    # 安全替换占位符，避免模板中的花括号与 str.format 冲突
    prompt = template
    prompt = prompt.replace("{metrics_json}", metrics_json)
    prompt = prompt.replace("{metrics_count}", str(len(compact_metrics)))
    prompt = prompt.replace("{output_text}", output_text)
    # 模板已移除 input_text，占位符不再使用
    return prompt


def normalize_scores(scored: Dict[str, Any], metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """根据指标校验与归一化模型输出：
    - 维度名称与指标对齐（支持规范化与模糊匹配，优先严格匹配）
    - 分数裁剪到 [0, max_score]
    - 重新计算加权总分 weighted_total
    """
    try:
        dims = scored.get("dimensions", [])
        if not isinstance(dims, list):
            dims = []

        # 指标映射与顺序表
        metric_map = {
            str(m.get("name", "")): {
                "max_score": float(m.get("max_score", 10.0)),
                "weight": float(m.get("weight", 1.0))
            }
            for m in metrics
        }

        def canonical(s: str) -> str:
            s = str(s).lower().strip()
            # 去除空白与常见标点，提高名称容错
            s = re.sub(r"[\s\u3000\-_.、，,。：；;！!？?`'\"\(\)\[\]\{\}]", "", s)
            return s

        # 规范化映射：直接同名、规范化同名
        metric_canon_map = {canonical(k): k for k in metric_map.keys()}

        normalized_by_name: Dict[str, Dict[str, Any]] = {}
        used_target_names = set()

        for d in dims:
            raw_name = str(d.get("name", "")).strip()
            # 1) 严格同名优先
            target = raw_name if raw_name in metric_map else None
            # 2) 规范化同名次之
            if target is None:
                c = canonical(raw_name)
                target = metric_canon_map.get(c)
            # 3) 模糊匹配（阈值），避免名称细微差异导致丢弃
            if target is None and raw_name:
                candidates = list(metric_map.keys())
                # 先对原始名称匹配
                best = max(
                    ((n, difflib.SequenceMatcher(None, raw_name, n).ratio()) for n in candidates),
                    key=lambda x: x[1],
                    default=(None, 0.0)
                )
                # 若原始名称匹配度不足，尝试规范化后匹配
                if best[1] < 0.75:
                    raw_c = canonical(raw_name)
                    best = max(
                        ((n, difflib.SequenceMatcher(None, raw_c, canonical(n)).ratio()) for n in candidates),
                        key=lambda x: x[1],
                        default=(None, 0.0)
                    )
                target = best[0] if best[1] >= 0.75 else None

            if target is None or target in used_target_names:
                # 无法对齐或已被其他维度使用，跳过此维度
                continue

            try:
                score = float(d.get("score", 0))
            except Exception:
                score = 0.0
            max_s = metric_map[target]["max_score"]
            weight = metric_map[target]["weight"]
            score = max(0.0, min(score, max_s))
            normalized_by_name[target] = {
                "name": target,
                "score": score,
                "max_score": max_s,
                "weight": weight,
                "reason": str(d.get("reason", ""))
            }
            used_target_names.add(target)

        # 二值指标纠偏：当“幻觉率”理由为无幻觉/无不合理时，若满分为1则置为1
        try:
            if "幻觉率" in normalized_by_name:
                nd = normalized_by_name["幻觉率"]
                max_s = float(nd.get("max_score", 1.0))
                if max_s <= 1.0:
                    reason_txt = str(nd.get("reason", ""))
                    positive = (
                        ("无幻觉" in reason_txt) or
                        ("未见幻觉" in reason_txt) or
                        ("未出现幻觉" in reason_txt) or
                        ("无不合理" in reason_txt) or
                        ("无不合理内容" in reason_txt) or
                        ("未见不合理" in reason_txt)
                    )
                    if positive and float(nd.get("score", 0.0)) < max_s:
                        nd["score"] = max_s
        except Exception:
            pass

        # 按指标顺序输出，缺失项补0分
        normalized = []
        for m in metrics:
            name = str(m.get("name", ""))
            if name in normalized_by_name:
                normalized.append(normalized_by_name[name])
            else:
                normalized.append({
                    "name": name,
                    "score": 0.0,
                    "max_score": float(m.get("max_score", 10.0)),
                    "weight": float(m.get("weight", 1.0)),
                    "reason": "未评估或缺失，默认0分"
                })

        # 重新计算加权总分
        weighted_total = 0.0
        for nd in normalized:
            if nd["max_score"] > 0:
                weighted_total += (nd["score"] / nd["max_score"]) * nd["weight"]
        return {"dimensions": normalized, "weighted_total": round(weighted_total, 6)}
    except Exception:
        # 出现异常时回退原始结果
        return scored


def main():
    cfg = load_config(os.path.join("config", "config.json"))
    cfg = apply_env_overrides(cfg)
    client = LocalLLMClient(cfg)
    metrics = []
    # 仅使用 Markdown 指标文件；若缺失则报错提示
    md_path = cfg.get("metrics_md_path")
    try:
        if md_path and os.path.exists(md_path):
            metrics = read_metrics_md(md_path)
    except Exception:
        metrics = []
    if not metrics:
        raise FileNotFoundError("未能加载评价指标，请确保存在 Markdown 指标文件并设置 metrics_md_path 或环境变量 METRICS_MD_PATH。")
    limit_metrics = int(cfg.get("limit_metrics", 0))
    if limit_metrics > 0:
        metrics = metrics[:limit_metrics]
    data = read_dataset(cfg["smallthink_json_path"])
    limit_raw = cfg.get("limit_samples", 0)
    try:
        if isinstance(limit_raw, str):
            val = limit_raw.strip().lower()
            if val in ("all", "全部", "全量", "full", "max"):
                limit = None
            else:
                limit = int(limit_raw)
        else:
            limit = int(limit_raw)
    except Exception:
        limit = None
    if isinstance(limit, int) and limit > 0:
        # 显式按顺序截取前 N 条样本，确保从第一个开始依次评估
        data = data[:limit]

    total = len(data)
    print(f"Starting sequential evaluation for {total} samples...")
    # 移除内存累计列表，仅进行逐样本评估与保存
    for idx, sample in enumerate(data):
        print(f"[{idx+1}/{total}] Evaluating sample index={idx} ...", flush=True)
        prompt = build_prompt(
            os.path.join("prompts", "eval_prompt_zh.txt"),
            metrics,
            sample,
            max_chars=int(cfg.get("input_truncate_chars", 2000))
        )
        # 尝试生成与解析，若维度为空或解析失败，进行一次重试
        scored = None
        scored_raw = None
        for attempt in range(2):
            try:
                scored_raw = client.generate_json(prompt)
            except Exception as e:
                scored_raw = {"dimensions": [], "weighted_total": 0.0, "raw": f"<error during generate_json: {e}>"}
            try:
                tmp = normalize_scores(scored_raw, metrics)
                # 有效结果：存在至少1个维度
                if isinstance(tmp.get("dimensions"), list) and len(tmp.get("dimensions")) > 0:
                    scored = tmp
                    break
                else:
                    print(f"  -> Retry due to empty dimensions (attempt {attempt+1})", flush=True)
            except Exception as e:
                print(f"  -> Retry due to normalize failure: {e} (attempt {attempt+1})", flush=True)
                scored = None
        # 若两次均失败，回退为零分结构（确保顺序与样本数量不受影响）
        if scored is None:
            scored = {
                "dimensions": [
                    {
                        "name": str(m.get("name", "")),
                        "score": 0.0,
                        "max_score": float(m.get("max_score", 10.0)),
                        "weight": float(m.get("weight", 1.0)),
                        "reason": "",
                    }
                    for m in metrics
                ],
                "weighted_total": 0.0,
            }
        # 不再累计到内存列表，直接进行单样本保存（见下方）

        # 单样本独立保存：以 数据集名_索引_scores.json 命名保存在 scores/ 目录
        try:
            os.makedirs("scores", exist_ok=True)
            dataset_base = os.path.splitext(os.path.basename(cfg.get("smallthink_json_path", "dataset.json")))[0]
            per_file = os.path.join("scores", f"{dataset_base}_{idx}_scores.json")
            tmp_per_file = per_file + ".tmp"
            per_payload = {
                "index": idx,
                "input": sample.get("input", ""),
                "output": sample.get("output") or sample.get("response") or sample.get("content") or "",
                "scores": scored,
                "model_raw": scored_raw.get("raw", ""),
            }
            with open(tmp_per_file, "w", encoding="utf-8") as pf:
                json.dump(per_payload, pf, ensure_ascii=False, indent=2)
            os.replace(tmp_per_file, per_file)
            print(f"Saved per-sample file -> {per_file}", flush=True)
        except Exception as e:
            print(f"Warning: failed to save per-sample file: {e}", flush=True)

    print(f"Completed sequential evaluation. Samples: {total}")


if __name__ == "__main__":
    main()
