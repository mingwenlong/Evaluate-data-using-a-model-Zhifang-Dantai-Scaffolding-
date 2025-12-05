import json
import re
import os
import fnmatch
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import threading
import time


class LocalLLMClient:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.mock_generation = bool(cfg.get("mock_generation", False))
        self.model_path = cfg.get("model_path")
        self.device_map = cfg.get("device_map", "auto")
        self.torch_dtype = cfg.get("torch_dtype", "auto")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.2))
        self.do_sample = bool(cfg.get("do_sample", False))
        self.top_p = float(cfg.get("top_p", 0.9))
        self.use_chat_template = bool(cfg.get("use_chat_template", True))
        self.max_input_tokens = int(cfg.get("max_input_tokens", 1024))

        # 若为模拟评估模式，跳过模型与分词器加载
        if self.mock_generation:
            self.tokenizer = None
            self.model = None
            return
        # 强制仅从本地目录加载，避免误触发远程 Hub 校验
        is_local_dir = isinstance(self.model_path, str) and os.path.isdir(self.model_path)
        local_only = bool(is_local_dir)

        # 预检查本地必要文件，给出更友好的报错
        if is_local_dir:
            expected_any_model = [
                os.path.join(self.model_path, "model.safetensors"),
                os.path.join(self.model_path, "pytorch_model.bin"),
            ]
            expected_any_tokenizer = [
                os.path.join(self.model_path, "tokenizer.json"),
                os.path.join(self.model_path, "tokenizer.model"),
                os.path.join(self.model_path, "vocab.json"),
            ]
            expected_config = os.path.join(self.model_path, "config.json")
            try:
                files = os.listdir(self.model_path)
            except Exception:
                files = []
            # 兼容分片权重与权重索引文件
            has_sharded_safetensors = any(fnmatch.fnmatch(f, "model-*.safetensors") for f in files)
            has_sharded_bin = any(fnmatch.fnmatch(f, "pytorch_model-*.bin") for f in files)
            has_weight_index = (
                os.path.exists(os.path.join(self.model_path, "model.safetensors.index.json")) or
                os.path.exists(os.path.join(self.model_path, "pytorch_model.bin.index.json"))
            )
            has_model = (
                any(os.path.exists(p) for p in expected_any_model) or
                has_sharded_safetensors or has_sharded_bin or has_weight_index
            )
            has_tok = any(os.path.exists(p) for p in expected_any_tokenizer)
            has_cfg = os.path.exists(expected_config)
            missing = []
            if not has_model:
                missing.append("model.safetensors/pytorch_model.bin 或 model-*.safetensors/pytorch_model-*.bin 及其 index.json")
            if not has_tok:
                missing.append("tokenizer.json/tokenizer.model/vocab.json")
            if not has_cfg:
                missing.append("config.json")
            if missing:
                raise FileNotFoundError(
                    (
                        f"本地模型目录缺少关键文件：{', '.join(missing)}。\n"
                        f"当前目录：{self.model_path}。\n"
                        "请使用 huggingface_hub.snapshot_download 下载完整文件，示例 allow_patterns：\n"
                        "[\"config.json\", \"tokenizer.json\", \"tokenizer_config.json\", \"model.safetensors\", \"model-*.safetensors\", \"pytorch_model.bin\", \"pytorch_model-*.bin\", \"model.safetensors.index.json\", \"pytorch_model.bin.index.json\", \"generation_config.json\", \"special_tokens_map.json\", \"tokenizer.model\", \"vocab.json\"]\n"
                        "或设置环境变量 MODEL_PATH 指向一个包含上述文件的本地目录。"
                    )
                )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        # 设备与精度选择：默认依赖 Transformers 的 device_map="auto" 在有GPU时放置到GPU
        use_cuda = torch.cuda.is_available()
        dtype_cfg = str(self.torch_dtype).lower()
        if use_cuda:
            if dtype_cfg in ("bfloat16", "bf16"):
                dtype = torch.bfloat16
            elif dtype_cfg in ("float16", "fp16"):
                dtype = torch.float16
            else:
                dtype = torch.bfloat16
        else:
            dtype = None  # CPU上用默认dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=local_only,
        )
        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        messages = [
            {"role": "system", "content": "你是一名严格的评审员，只输出严格的JSON对象。"},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_json(self, prompt: str) -> Dict:
        # 从提示中预解析指标映射：name -> {max_score, weight}
        metric_map = {}
        try:
            for m in re.finditer(r"\[[\s\S]*?\]", prompt):
                try:
                    arr = json.loads(m.group(0))
                    if isinstance(arr, list) and arr and all(isinstance(x, dict) for x in arr):
                        if all(("name" in x and "weight" in x and "max_score" in x) for x in arr):
                            for it in arr:
                                name = str(it.get("name", ""))
                                if name:
                                    metric_map[name] = {
                                        "max_score": float(it.get("max_score", 10.0)),
                                        "weight": float(it.get("weight", 1.0)),
                                    }
                            break
                except Exception:
                    continue
        except Exception:
            metric_map = {}
        # 模拟评估模式：从提示中解析指标JSON（name/weight/max_score），并生成对齐的非零分
        if self.mock_generation:
            try:
                metrics_list = []
                # 遍历所有可能的 JSON 数组片段，筛选包含 name/weight/max_score 的指标列表
                for m in re.finditer(r"\[[\s\S]*?\]", prompt):
                    try:
                        arr = json.loads(m.group(0))
                        if isinstance(arr, list) and arr and all(isinstance(x, dict) for x in arr):
                            # 校验至少存在 name/weight/max_score 三列
                            if all(("name" in x and "weight" in x and "max_score" in x) for x in arr):
                                metrics_list = [
                                    {
                                        "name": str(it.get("name", "")),
                                        "weight": float(it.get("weight", 1.0)),
                                        "max_score": float(it.get("max_score", 10.0)),
                                    }
                                    for it in arr
                                ]
                                break
                    except Exception:
                        continue
                if not metrics_list:
                    return {"dimensions": [], "weighted_total": 0.0, "raw": ""}
                dims = []
                for i, mi in enumerate(metrics_list):
                    max_s = mi["max_score"]
                    # 二值或小满分维度给出 1 或 0；连续型给出 60%~80% 区间的分数
                    if max_s <= 1.0:
                        score = 1.0 if (i % 2 == 0) else 0.0
                    else:
                        score = round(max_s * (0.6 + 0.1 * (i % 3)), 3)
                    dims.append({
                        "name": mi["name"],
                        "score": max(0.0, min(score, max_s)),
                        "max_score": max_s,
                        "weight": mi["weight"],
                        "reason": "模拟评估：管线校验"
                    })
                wt = 0.0
                for d in dims:
                    if d["max_score"] > 0:
                        wt += (d["score"] / d["max_score"]) * d["weight"]
                raw_json = json.dumps({"dimensions": dims, "weighted_total": round(wt, 6)}, ensure_ascii=False)
                return {"dimensions": dims, "weighted_total": round(wt, 6), "raw": raw_json}
            except Exception:
                return {"dimensions": [], "weighted_total": 0.0, "raw": ""}
        text = self._format_prompt(prompt)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        # 将输入移动到合适设备（单GPU或CPU）。若模型为多设备分片，通常为单GPU即可。
        in_device = getattr(self.model, "device", None)
        if in_device is None:
            in_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(in_device)
        # 早停：检测到完整JSON闭合或超时则停止（仅基于生成的新tokens判断）
        start_time = time.time()
        max_time = float(self.cfg.get("max_gen_time_seconds", 20))
        input_len = int(inputs["input_ids"].shape[1])

        # 从提示中提取期望的维度数量（metrics_json 的长度），用于判断是否生成完整
        def _metrics_expected_count(prompt_text: str) -> int:
            # 优先解析模板中显式标记的维度总数，如："维度总数为 N"
            try:
                mcount = re.search(r"维度总数为\s*(\d+)", prompt_text)
                if mcount:
                    return int(mcount.group(1))
            except Exception:
                pass
            # 回退：解析 metrics_json 的对象数组长度
            try:
                for m in re.finditer(r"\[[\s\S]*?\]", prompt_text):
                    try:
                        arr = json.loads(m.group(0))
                        if isinstance(arr, list) and arr and all(isinstance(x, dict) for x in arr):
                            # 以包含 name 的对象数组作为指标列表
                            if all(("name" in x) for x in arr):
                                return len(arr)
                    except Exception:
                        continue
            except Exception:
                pass
            return 0

        expected_count = _metrics_expected_count(prompt)

        class JsonStop(StoppingCriteria):
            def __init__(self, tokenizer, prompt_token_count: int, expected_count: int):
                self.buf = ""
                self.tokenizer = tokenizer
                self.prompt_token_count = prompt_token_count
                self.expected_count = int(expected_count or 0)
            def __call__(self, input_ids, scores, **kwargs):
                try:
                    gen_only_ids = input_ids[0][self.prompt_token_count:]
                    text = self.tokenizer.decode(gen_only_ids, skip_special_tokens=True)
                except Exception:
                    text = ""
                self.buf = text
                if time.time() - start_time > max_time:
                    return True
                # 仅在生成出包含 "dimensions" 键的 JSON 对象时早停，避免误捕提示中的指标 JSON 片段
                s = text.find("{")
                e = text.rfind("}")
                if s != -1 and e != -1 and e > s:
                    frag = text[s:e+1]
                    try:
                        obj = json.loads(frag)
                        # 完整对象要求：包含 dimensions（列表或字典），维度数达到期望数量
                        if isinstance(obj, dict) and obj.get("dimensions") is not None:
                            dims_val = obj.get("dimensions")
                            if isinstance(dims_val, list):
                                enough = (self.expected_count <= 0) or (len(dims_val) >= self.expected_count)
                                fields_ok = all(isinstance(d, dict) and ("name" in d and "score" in d) for d in dims_val)
                                if enough and fields_ok:
                                    return True
                            elif isinstance(dims_val, dict):
                                # 同时支持值为对象或数值（如 {name: 85.0}），忽略内部 weighted_total
                                count = 0
                                for k, v in dims_val.items():
                                    if str(k) == "weighted_total":
                                        continue
                                    if isinstance(v, dict) or isinstance(v, (int, float)):
                                        count += 1
                                enough = (self.expected_count <= 0) or (count >= self.expected_count)
                                if enough:
                                    return True
                        return False
                    except Exception:
                        return False
                return False

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        stop_criteria = StoppingCriteriaList([JsonStop(self.tokenizer, input_len, expected_count)])
        def _gen():
            with torch.inference_mode():
                gen_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "do_sample": self.do_sample,
                    "stopping_criteria": stop_criteria,
                    "streamer": streamer,
                    "max_time": max_time,
                }
                # 仅在采样时传入温度与 top_p，避免无效参数警告
                if self.do_sample:
                    gen_kwargs["temperature"] = self.temperature
                    gen_kwargs["top_p"] = self.top_p
                # EOS/PAD 回退处理
                eos_id = self.tokenizer.eos_token_id
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id
                if eos_id is not None:
                    gen_kwargs["eos_token_id"] = eos_id
                if pad_id is not None:
                    gen_kwargs["pad_token_id"] = pad_id
                self.model.generate(**inputs, **gen_kwargs)
        th = threading.Thread(target=_gen, daemon=True)
        th.start()
        buf = ""
        def _extract_json_with_dimensions(text: str) -> str:
            # 在生成文本中寻找包含 "dimensions" 键的 JSON 对象片段
            # 从最后一个 "}" 开始向前回溯匹配最早的 "{"，尝试解析为对象并检查键
            idx_end = text.rfind("}")
            while idx_end != -1:
                idx_start = text.find("{")
                if idx_start == -1 or idx_start >= idx_end:
                    break
                frag = text[idx_start:idx_end+1]
                try:
                    obj = json.loads(frag)
                    # 要求包含 dimensions（数组或字典），维度数量达到期望；不强制 weighted_total
                    if isinstance(obj, dict) and obj.get("dimensions") is not None:
                        dims_val = obj.get("dimensions")
                        if isinstance(dims_val, list):
                            enough = (expected_count <= 0) or (len(dims_val) >= expected_count)
                            if enough:
                                return frag
                        elif isinstance(dims_val, dict):
                            # 支持值为对象或数值，忽略内部 weighted_total
                            count = 0
                            for k, v in dims_val.items():
                                if str(k) == "weighted_total":
                                    continue
                                if isinstance(v, dict) or isinstance(v, (int, float)):
                                    count += 1
                            enough = (expected_count <= 0) or (count >= expected_count)
                            if enough:
                                return frag
                except Exception:
                    pass
                # 去掉末尾一个字符继续尝试
                idx_end = text.rfind("}", 0, idx_end)
            return ""

        for piece in streamer:
            buf += piece
            if time.time() - start_time > max_time:
                break
            frag_try = _extract_json_with_dimensions(buf)
            if frag_try:
                buf = frag_try
                break
        out = buf
        if not out:
            # 回退：返回最小结构，留空原始片段
            return {"dimensions": [], "weighted_total": 0.0, "raw": out}
        frag = out
        # 轻度清洗：修复双花括号与尾逗号，尽量让近似JSON可解析
        def _clean_json_like(text: str) -> str:
            s = text
            # 合并连续的花括号，如 '{{' -> '{'，'}}' -> '}'
            s = re.sub(r"\{\{+", "{", s)
            s = re.sub(r"\}\}+", "}", s)
            # 去除对象/数组中的非法尾逗号
            s = re.sub(r",\s*\}", "}", s)
            s = re.sub(r",\s*\]", "]", s)
            # 去除Markdown代码块围栏与反引号
            s = re.sub(r"```+\s*json\s*", "", s, flags=re.IGNORECASE)
            s = re.sub(r"```+", "", s)
            s = s.replace("`", "")
            # 若包含前后多余文本，仅保留最外层大括号包围的片段
            try:
                si = s.find("{")
                ei = s.rfind("}")
                if si != -1 and ei != -1 and ei > si:
                    s = s[si:ei+1]
            except Exception:
                pass
            # 去除在维度对象中的多余键，确保最简结构
            # 将 reason 以外的未知键尝试剔除（仅在简单正则可行时）
            return s
        cleaned = _clean_json_like(frag)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                # 归一化 dimensions：支持字典 -> 列表，并补齐缺失的 max_score / weight
                dims_val = parsed.get("dimensions")
                inner_total = None
                if isinstance(dims_val, dict):
                    # 维度字典中可能夹带 weighted_total，提升为顶层
                    wt_in = dims_val.get("weighted_total")
                    if isinstance(wt_in, (int, float)):
                        inner_total = wt_in
                    dims_list = []
                    for name, info in dims_val.items():
                        if name == "weighted_total":
                            continue
                        # 同时支持嵌套对象或数值（如 {name: 85.0}）
                        if isinstance(info, dict):
                            try:
                                score = float(info.get("score", 0))
                            except Exception:
                                score = 0.0
                            reason = str(info.get("reason", ""))
                            mm = metric_map.get(str(name), {"max_score": 10.0, "weight": 1.0})
                            dims_list.append({
                                "name": str(name),
                                "score": score,
                                "max_score": float(info.get("max_score", mm["max_score"])),
                                "weight": float(info.get("weight", mm["weight"])),
                                "reason": reason,
                            })
                        elif isinstance(info, (int, float)):
                            mm = metric_map.get(str(name), {"max_score": 10.0, "weight": 1.0})
                            dims_list.append({
                                "name": str(name),
                                "score": float(info),
                                "max_score": float(mm["max_score"]),
                                "weight": float(mm["weight"]),
                                "reason": "",
                            })
                    parsed["dimensions"] = dims_list
                elif isinstance(dims_val, list):
                    dims_list = []
                    for d in dims_val:
                        if not isinstance(d, dict):
                            continue
                        name = str(d.get("name", ""))
                        try:
                            score = float(d.get("score", 0))
                        except Exception:
                            score = 0.0
                        mm = metric_map.get(name, {"max_score": 10.0, "weight": 1.0})
                        dims_list.append({
                            "name": name,
                            "score": score,
                            "max_score": float(d.get("max_score", mm["max_score"])),
                            "weight": float(d.get("weight", mm["weight"])),
                            "reason": str(d.get("reason", "")),
                        })
                    parsed["dimensions"] = dims_list
                # 若缺失 weighted_total 或存在于维度字典中，则计算或提升
                if not isinstance(parsed.get("weighted_total"), (int, float)):
                    if isinstance(inner_total, (int, float)):
                        parsed["weighted_total"] = float(inner_total)
                    else:
                        try:
                            wt = 0.0
                            dims_calc = parsed.get("dimensions", []) if isinstance(parsed.get("dimensions"), list) else []
                            for d in dims_calc:
                                ms = float(d.get("max_score", 0) or 0)
                                if ms > 0:
                                    wt += (float(d.get("score", 0) or 0) / ms) * float(d.get("weight", 1) or 1)
                            parsed["weighted_total"] = round(wt, 6)
                        except Exception:
                            parsed["weighted_total"] = 0.0
                parsed["raw"] = frag
                return parsed
        except Exception:
            # 兜底：从原始片段中提取包含 name/score/max_score/weight 的对象列表
            dims = []
            for mm in re.finditer(r"\{[\s\S]*?\}", frag):
                obj_txt = _clean_json_like(mm.group(0))
                try:
                    obj = json.loads(obj_txt)
                except Exception:
                    # 尝试通过字段级正则抽取
                    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', obj_txt)
                    score_m = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', obj_txt)
                    max_m = re.search(r'"max_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', obj_txt)
                    weight_m = re.search(r'"weight"\s*:\s*([0-9]+(?:\.[0-9]+)?)', obj_txt)
                    reason_m = re.search(r'"reason"\s*:\s*"([^"]*)"', obj_txt)
                    if name_m and score_m and max_m and weight_m:
                        try:
                            obj = {
                                "name": name_m.group(1),
                                "score": float(score_m.group(1)),
                                "max_score": float(max_m.group(1)),
                                "weight": float(weight_m.group(1)),
                                "reason": reason_m.group(1) if reason_m else "",
                            }
                        except Exception:
                            obj = None
                    else:
                        obj = None
                if isinstance(obj, dict) and all(k in obj for k in ("name", "score", "max_score", "weight")):
                    dims.append(obj)
            if dims:
                wt = 0.0
                for d in dims:
                    if d.get("max_score", 0) > 0:
                        wt += (float(d.get("score", 0)) / float(d.get("max_score", 1))) * float(d.get("weight", 1))
                return {"dimensions": dims, "weighted_total": round(wt, 6), "raw": frag}
            # 最后回退：保留原始片段，维度空
            return {"dimensions": [], "weighted_total": 0.0, "raw": frag}
