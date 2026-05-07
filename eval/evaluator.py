"""VIBEResearch evaluation: knowledge graph construction from multi-step queries.

Data: directory of task_*.json files, each with final_query (input) and nodes+triples (ground truth).
Evaluation: LLM-as-judge with GT coverage (recall), pred validity (precision), and derived node metrics.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .grader import create_grader, GraderClient

logger = logging.getLogger(__name__)

NUM_EVAL_BATCHES = 5
_COUNT_RETRIES = 3


def _index_results(items: List[dict], expected: int, offset: int = 0) -> Optional[List[dict]]:
    """Try to build a positional list from items with 'index' fields.

    Returns a list of length *expected* if every index offset..offset+expected-1
    is present exactly once; otherwise returns None.
    """
    by_idx: Dict[int, dict] = {}
    for item in items:
        idx = item.get("index")
        if idx is None:
            return None
        if not isinstance(idx, int) or idx < offset or idx >= offset + expected:
            continue
        by_idx.setdefault(idx, item)
    if len(by_idx) == expected:
        return [by_idx[i] for i in range(offset, offset + expected)]
    return None


def _index_results_best_effort(
    items: List[dict], expected: int, default: dict, offset: int = 0,
) -> List[dict]:
    """Best-effort index-based alignment with fallback to positional + padding."""
    by_idx: Dict[int, dict] = {}
    for item in items:
        idx = item.get("index")
        if isinstance(idx, int) and offset <= idx < offset + expected:
            by_idx.setdefault(idx, item)
    if by_idx:
        return [by_idx.get(i, default) for i in range(offset, offset + expected)]
    # No index fields — fall back to positional
    return (items[:expected] + [default] * expected)[:expected]


# ---------------------------------------------------------------------------
# LLM helpers (use grader backend for judge calls)
# ---------------------------------------------------------------------------


def _llm_json(grader: GraderClient, prompt: str) -> Any:
    """Call grader LLM and parse JSON response."""
    text = grader.complete(prompt).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Node metrics (derived from triple matching results)
# ---------------------------------------------------------------------------


def _derive_node_metrics(
    pred_triples: List[dict],
    gt_data: dict,
    coverage_res: dict,
) -> dict:
    """Derive node-level metrics from coverage results only.

    - Node Recall: fraction of GT nodes that appear in at least one covered GT triple.
    - Node Precision: fraction of pred nodes that appear in at least one supporting pred triple.
    """
    id_to_name = {n["node_id"]: n["node_name"] for n in gt_data.get("nodes", [])}
    gt_triples = gt_data.get("triples", [])

    all_gt_nodes = {id_to_name.get(n["node_id"], n["node_id"]) for n in gt_data.get("nodes", [])}
    all_pred_nodes = {
        str(n) for t in pred_triples for n in (t.get("head", ""), t.get("tail", "")) if n
    }

    covered_gt_nodes = set()
    supporting_pred_indices = set()
    for item in coverage_res.get("per_gt_triple", []):
        if item["covered"]:
            gt_idx = item["gt_index"]
            if gt_idx < len(gt_triples):
                t = gt_triples[gt_idx]
                covered_gt_nodes.add(id_to_name.get(t["head_id"], t["head_id"]))
                covered_gt_nodes.add(id_to_name.get(t["tail_id"], t["tail_id"]))
        for idx in item.get("supporting_pred_indices", []):
            supporting_pred_indices.add(idx)

    precise_pred_nodes = set()
    for idx in supporting_pred_indices:
        if idx < len(pred_triples):
            pt = pred_triples[idx]
            if pt.get("head"):
                precise_pred_nodes.add(str(pt["head"]))
            if pt.get("tail"):
                precise_pred_nodes.add(str(pt["tail"]))

    recalled = len(covered_gt_nodes & all_gt_nodes)
    total_gt = len(all_gt_nodes)
    precise = len(precise_pred_nodes & all_pred_nodes)
    total_pred = len(all_pred_nodes)

    r = recalled / total_gt if total_gt else 0.0
    p = precise / total_pred if total_pred else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    return {
        "total_gt_nodes": total_gt,
        "recalled_gt_nodes": recalled,
        "total_pred_nodes": total_pred,
        "precise_pred_nodes": precise,
        "node_precision": round(p, 4),
        "node_recall": round(r, 4),
        "node_f1": round(f1, 4),
    }


# ---------------------------------------------------------------------------
# Phase 2 & 3: Information entailment evaluation
# ---------------------------------------------------------------------------


def _split_into_n_batches(lst: list, n: int = NUM_EVAL_BATCHES) -> List[list]:
    """Split *lst* into at most *n* batches of roughly equal size."""
    if not lst:
        return []
    n = min(n, len(lst))
    k, m = divmod(len(lst), n)
    batches = []
    idx = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        batches.append(lst[idx : idx + size])
        idx += size
    return batches


def _format_pred_triples(pred_triples: List[dict]) -> str:
    """Format predicted triples as a numbered list for the LLM judge (P-prefix)."""
    lines = []
    for i, t in enumerate(pred_triples):
        lines.append(f'P-{i}: "{t.get("head", "")}" --[{t.get("relation", "")}]--> "{t.get("tail", "")}"')
    return "\n".join(lines)



def _gt_triple_to_dict(t: dict, gt_data: dict) -> dict:
    """Convert a raw GT triple (with node IDs) to {head, relation, tail} with names."""
    id_to_name = {n["node_id"]: n["node_name"] for n in gt_data.get("nodes", [])}
    return {
        "head": id_to_name.get(t["head_id"], t["head_id"]),
        "relation": t["relation"],
        "tail": id_to_name.get(t["tail_id"], t["tail_id"]),
    }


def _run_gt_coverage_batch(
    grader: GraderClient,
    batch: List[dict],
    batch_offset: int,
    pred_ctx: str,
    gt_data: dict,
) -> List[dict]:
    """Judge one batch of GT triples for coverage. Returns per-GT-triple details."""
    id_to_name = {n["node_id"]: n["node_name"] for n in gt_data.get("nodes", [])}
    claims = []
    for i, t in enumerate(batch):
        global_idx = batch_offset + i
        h = id_to_name.get(t["head_id"], t["head_id"])
        tail = id_to_name.get(t["tail_id"], t["tail_id"])
        claims.append(f'GT-{global_idx}: "{h}" --[{t["relation"]}]--> "{tail}"')
    claims_text = "\n".join(claims)

    first_idx = batch_offset
    last_idx = batch_offset + len(batch) - 1

    prompt = (
        "你是知识图谱评测专家。给定一组预测的知识图谱三元组（P-编号），判断每个 ground truth 声明（GT-编号）是否被预测图谱的信息所**覆盖**。\n\n"
        f"## 预测图谱三元组（P-编号）\n{pred_ctx}\n\n"
        f"## 待判断的 Ground Truth 声明（共 {len(batch)} 条，编号 GT-{first_idx}~GT-{last_idx}）\n{claims_text}\n\n"
        "## 判断标准\n"
        "一个 GT 声明算\"被覆盖\"当且仅当预测图谱中的信息**蕴含了该 GT 声明的全部事实**：\n"
        "1. 预测图谱中有一条三元组直接表达了相同信息（允许实体别名、关系同义词），或\n"
        "2. 预测图谱中某条三元组的信息量更大，包含了该 GT 声明作为子信息（如预测列出所有成分，GT 只列其中一个），或\n"
        "3. 预测图谱中多条三元组合起来完整覆盖了该 GT 声明的信息（如预测分别列出各个子部分，GT 合并为一条），或\n"
        "4. 预测图谱中多条三元组**通过图谱中已有的显式关系**（如包含、属于、位于、别名）可以组合起来覆盖该 GT 声明\n\n"
        "不覆盖：预测图谱中完全找不到对应信息，或预测信息与 GT 矛盾\n\n"

        "## 常见错误（必须避免）\n"
        "- **无关实体不算覆盖**：如果预测图谱中没有建立两个实体之间的关系，不能将一个实体的属性当作另一个实体的属性\n"
        "- **近义但不等价不算覆盖**：如\"气急\"和\"呼吸困难\"是不同症状，不能混为一谈\n"
        "- **需要外部知识的推理不算覆盖**：仅凭预测图谱中的信息无法得出、需要额外背景知识才能推出的结论不算覆盖\n"
        "- **编号不要混淆**：supporting_pred_indices 填的是 P-X 的数字，不是 GT-X 的数字\n\n"
        "## 示例\n"
        "假设预测图谱中有：\n"
        '  P-10: "北京大学" --[位置]--> "北京市海淀区"\n'
        '  P-11: "过敏性哮喘" --[属于]--> "春季过敏性疾病"\n'
        '  P-12: "过敏性哮喘" --[典型症状]--> "反复干咳、喘息、胸闷、呼吸困难"\n'
        '  P-13: "Constance Moofushi" --[餐饮计划]--> "一价全包"\n'
        '  P-14: "Kunfunadhoo" --[包含]--> "Soneva Fushi"\n'
        '  P-15: "Kunfunadhoo" --[交通方式]--> "水上飞机约30分钟"\n\n'
        "示例判断：\n"
        '- GT: "北京大学" --[所在城市]--> "北京"  → covered=true, supporting=[P-10]  ✓ 直接表达相同信息\n'
        '- GT: "Soneva Fushi" --[交通方式]--> "水飞约30分钟"  → covered=true, supporting=[P-14, P-15]  ✓ P-14显式说明Kunfunadhoo包含Soneva Fushi，P-15说明Kunfunadhoo交通方式是水上飞机约30分钟，组合起来覆盖\n'
        '- GT: "春季高发过敏性疾病" --[常见表现]--> "气急"  → covered=false  ✓ P-11和P-12说的是"过敏性哮喘"的症状，不是"春季高发过敏性疾病"整体的；且"呼吸困难"≠"气急"\n'
        '- GT: "Waldorf Astoria" --[是否提供全包]--> "不提供"  → covered=false  ✓ P-13说的是Constance Moofushi，与Waldorf Astoria无关\n\n'
        "## 输出格式\n"
        f"results 数组必须恰好包含 {len(batch)} 个元素。每个元素的 index 字段必须是对应 GT 声明的编号（即 GT-X 中的数字 X，范围 {first_idx}~{last_idx}）。\n"
        "supporting_pred_indices 中填写支持该 GT 声明的预测三元组编号（即 P-X 中的数字 X）。\n"
        "如果不确定是否覆盖，请判为 false。\n\n"
        '输出 JSON：\n{"results": [{"index": ' + str(first_idx) + ', "covered": true, "supporting_pred_indices": [3, 7]}, '
        '{"index": ' + str(first_idx + 1) + ', "covered": false, "supporting_pred_indices": []}, ..., {"index": ' + str(last_idx) + ', ...}]}\n'
        "只输出 JSON，不要其他内容。"
    )

    items = None
    for attempt in range(_COUNT_RETRIES):
        try:
            raw_text = grader.complete(prompt).strip()
            logger.debug("GT coverage raw response (attempt %d, batch=%d): %s",
                         attempt + 1, len(batch), raw_text[:500])
            text = raw_text
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            result = json.loads(text)
            cur = result.get("results", []) if isinstance(result, dict) else []
            indexed = _index_results(cur, len(batch), offset=batch_offset)
            if indexed is not None:
                items = indexed
                break
            logger.warning("GT coverage batch returned %d items, expected %d (attempt %d/%d)",
                           len(cur), len(batch), attempt + 1, _COUNT_RETRIES)
            items = _index_results_best_effort(cur, len(batch), {"covered": False, "supporting_pred_indices": []}, offset=batch_offset)
        except Exception as e:
            logger.warning("GT coverage batch failed (attempt %d/%d): %s", attempt + 1, _COUNT_RETRIES, e)
    if items is None:
        items = [{"covered": False, "supporting_pred_indices": []}] * len(batch)

    results = []
    for i, (t, item) in enumerate(zip(batch, items)):
        results.append({
            "gt_index": batch_offset + i,
            "gt_triple": _gt_triple_to_dict(t, gt_data),
            "covered": bool(item.get("covered", False)),
            "supporting_pred_indices": item.get("supporting_pred_indices", []),
        })
    return results


def _evaluate_gt_coverage(
    grader: GraderClient,
    pred_triples: List[dict],
    gt_data: dict,
) -> dict:
    """For each GT triple, check whether the pred graph entails its information.

    Batches are evaluated in parallel. Returns recall and per-GT-triple details.
    """
    gt_triples = gt_data.get("triples", [])
    if not gt_triples or not pred_triples:
        return {
            "total_gt_triples": len(gt_triples),
            "covered_count": 0,
            "triplet_recall": 0.0,
            "per_gt_triple": [],
        }

    pred_ctx = _format_pred_triples(pred_triples)
    batches = _split_into_n_batches(gt_triples, NUM_EVAL_BATCHES)

    offsets = []
    idx = 0
    for b in batches:
        offsets.append(idx)
        idx += len(b)

    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        futures = {
            pool.submit(_run_gt_coverage_batch, grader, batch, offset, pred_ctx, gt_data): bi
            for bi, (batch, offset) in enumerate(zip(batches, offsets))
        }
        batch_results = [None] * len(batches)
        for fut in as_completed(futures):
            batch_results[futures[fut]] = fut.result()

    per_gt = []
    for br in batch_results:
        per_gt.extend(br)

    covered = sum(1 for g in per_gt if g["covered"])
    recall = covered / len(gt_triples) if gt_triples else 0.0

    return {
        "total_gt_triples": len(gt_triples),
        "covered_count": covered,
        "triplet_recall": round(recall, 4),
        "per_gt_triple": per_gt,
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_triples(text: str) -> List[dict]:
    """Extract triple list from agent response text."""

    def _normalize(triples: list) -> List[dict]:
        """Ensure head/relation/tail are all strings."""
        return [
            {k: str(v) if v is not None else "" for k, v in t.items()}
            for t in triples if isinstance(t, dict)
        ]

    def _try_parse(s: str) -> Optional[list]:
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    def _repair_and_parse(s: str) -> Optional[list]:
        """Try to fix common JSON issues: curly quotes, control chars, etc."""
        import re as _re
        repaired = s
        repaired = repaired.replace("“", r"\"").replace("”", r"\"")
        repaired = repaired.replace("‘", r"\'").replace("’", r"\'")
        repaired = _re.sub(r"[\x00-\x1f]", lambda m: " " if m.group() not in "\n\r\t" else m.group(), repaired)
        result = _try_parse(repaired)
        if result is not None:
            return result
        # Last resort: extract individual triple objects via regex
        triples = []
        for m in _re.finditer(
            r'\{\s*"head"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"relation"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"tail"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}',
            s,
        ):
            triples.append({"head": m.group(1), "relation": m.group(2), "tail": m.group(3)})
        return triples if triples else None

    text = text.strip()
    # Try to find JSON array in markdown code block
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            result = _try_parse(block)
            if result is not None:
                return _normalize(result)
        # Retry with repair on the json block
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if not block:
                continue
            result = _repair_and_parse(block)
            if result:
                return _normalize(result)
    # Try direct JSON parse
    result = _try_parse(text)
    if result is not None:
        return _normalize(result)
    # Try to find array substring
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        substr = text[start : end + 1]
        result = _try_parse(substr)
        if result is not None:
            return _normalize(result)
        result = _repair_and_parse(substr)
        if result:
            return _normalize(result)
    return []


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load task JSON files from data_path directory.

    Each task JSON should have: task_id, final_query, nodes, triples, language.
    """
    if os.path.isfile(data_path):
        files = [data_path]
    else:
        files = sorted(
            f for f in (os.path.join(data_path, n) for n in os.listdir(data_path))
            if f.endswith(".json")
        )
    data = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            task = json.load(f)
        question = task["final_query"]
        data.append({
            "qid": task["task_id"],
            "question": question,
            "initial_query": task.get("initial_query", question),
            "sub_queries": task.get("sub_queries", []),
            "user_queries": task.get("user_queries", []),
            "user_persona": task.get("user_persona", ""),
            "answer": {"nodes": task["nodes"], "triples": task["triples"]},
            "language": task.get("language", "en"),
        })
        if limit and len(data) >= limit:
            break
    logger.info("VIBEResearch loaded %d tasks from %s", len(data), data_path)
    return data


# ---------------------------------------------------------------------------
# Grading: single-item
# ---------------------------------------------------------------------------


def grade_one(
    grader: GraderClient,
    qid: str,
    sample_idx: int,
    response: str,
    gt_data: dict,
) -> dict:
    """Grade a single trajectory's response against ground truth.

    Uses information entailment evaluation:
    - GT coverage (recall): for each GT triple, does the pred graph entail it?
    - Pred validity (precision): for each pred triple, does the GT graph support it?
    """
    pred_triples = parse_triples(response)
    result = {"qid": qid, "sample_idx": sample_idx}

    if not pred_triples:
        result.update({
            "node_precision": 0.0, "node_recall": 0.0, "node_f1": 0.0,
            "triplet_precision": 0.0, "triplet_recall": 0.0, "triplet_f1": 0.0,
            "num_pred_triples": 0,
        })
        return result

    coverage_res = _evaluate_gt_coverage(grader, pred_triples, gt_data)

    # Precision: unique pred indices referenced in supporting_pred_indices / total pred
    supporting_indices = set()
    for item in coverage_res.get("per_gt_triple", []):
        for idx in item.get("supporting_pred_indices", []):
            supporting_indices.add(idx)
    precision = len(supporting_indices) / len(pred_triples) if pred_triples else 0.0

    recall = coverage_res["triplet_recall"]
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    node_res = _derive_node_metrics(pred_triples, gt_data, coverage_res)

    result.update({
        "node_precision": node_res["node_precision"],
        "node_recall": node_res["node_recall"],
        "node_f1": node_res["node_f1"],
        "node_details": node_res,
        "triplet_precision": round(precision, 4),
        "triplet_recall": recall,
        "triplet_f1": round(f1, 4),
        "coverage_details": coverage_res,
        "num_pred_triples": len(pred_triples),
    })
    return result


# ---------------------------------------------------------------------------
# Grading: batch
# ---------------------------------------------------------------------------


def grade(
    items: List[Dict[str, Any]],
    grader_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Grade all items and return summary metrics.

    Parameters
    ----------
    items : list
        Each item has qid, answer (ground truth), and responses (list of response strings).
    grader_config : dict
        Config for creating a GraderClient.

    Returns
    -------
    dict with per-query aggregate metrics (avg@N, best@N), per_item list, and summary.
    """
    grader = create_grader(grader_config)
    max_workers = grader_config.get("max_workers", 4)

    # Flatten items x samples
    flat = []
    for item in items:
        responses = item.get("responses", [item.get("response", "")])
        for si, resp in enumerate(responses):
            flat.append({**item, "response": resp, "sample_idx": si})

    def _grade_one_wrapper(fi: dict) -> dict:
        return grade_one(
            grader,
            fi["qid"],
            fi.get("sample_idx", 0),
            fi.get("response", ""),
            fi["answer"],
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        per_item = list(tqdm(pool.map(_grade_one_wrapper, flat), total=len(flat), desc="VIBEResearch grading"))

    # Group by qid
    metrics_keys = ["node_precision", "node_recall", "node_f1",
                    "triplet_precision", "triplet_recall", "triplet_f1"]
    by_qid = {}
    for r in per_item:
        by_qid.setdefault(r["qid"], []).append(r)

    num_samples = max((len(v) for v in by_qid.values()), default=1)

    # Per-query: avg and best (by triplet_f1) across samples
    avg_vals = {k: [] for k in metrics_keys}
    best_vals = {k: [] for k in metrics_keys}
    for qid, samples in by_qid.items():
        for k in metrics_keys:
            sv = [s[k] for s in samples]
            avg_vals[k].append(sum(sv) / len(sv))
        best = max(samples, key=lambda s: s["triplet_f1"])
        for k in metrics_keys:
            best_vals[k].append(best[k])

    summary = {}
    for k in metrics_keys:
        summary[f"avg@{num_samples}_{k}"] = round(sum(avg_vals[k]) / len(avg_vals[k]), 4) if avg_vals[k] else 0.0
        summary[f"best@{num_samples}_{k}"] = round(sum(best_vals[k]) / len(best_vals[k]), 4) if best_vals[k] else 0.0
    summary["num_queries"] = len(by_qid)
    summary["num_samples"] = num_samples
    summary["per_item"] = per_item
    return summary
