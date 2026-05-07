"""
生成 meta-evaluation 的 CSV 文件，供人工质检。

对每个 task_xxx.json，生成两个 CSV：
  - gt 文件夹: 每个 GT 三元组 + 模型判断是否被 pred 覆盖 + 支撑的 pred 三元组文字
  - pred 文件夹: 每个 pred 三元组 + 模型判断是否有效 + 支撑的 GT 三元组文字
"""

import json
import csv
import os
import glob
import argparse


def format_triple(t):
    return f"({t['head']}, {t['relation']}, {t['tail']})"


def process_one_file(json_path, gt_out_dir, pred_out_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    basename = os.path.splitext(os.path.basename(json_path))[0]

    coverage = data["coverage_details"]
    validity = data["validity_details"]

    pred_triples = {
        item["pred_index"]: item["pred_triple"]
        for item in validity["per_pred_triple"]
    }
    gt_triples = {
        item["gt_index"]: item["gt_triple"]
        for item in coverage["per_gt_triple"]
    }

    # --- GT CSV (recall check) ---
    gt_csv_path = os.path.join(gt_out_dir, f"{basename}.csv")
    with open(gt_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gt_index", "head", "relation", "tail",
            "model_covered", "supporting_pred_triples",
            "human_label", "human_comment",
        ])
        for item in coverage["per_gt_triple"]:
            gt = item["gt_triple"]
            support_strs = []
            for idx in item["supporting_pred_indices"]:
                if idx in pred_triples:
                    support_strs.append(format_triple(pred_triples[idx]))
            writer.writerow([
                item["gt_index"],
                gt["head"],
                gt["relation"],
                gt["tail"],
                "是" if item["covered"] else "否",
                "\n".join(support_strs) if support_strs else "",
                "",
                "",
            ])

    # --- Pred CSV (precision check) ---
    pred_csv_path = os.path.join(pred_out_dir, f"{basename}.csv")
    with open(pred_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "pred_index", "head", "relation", "tail",
            "model_valid", "supporting_gt_triples",
            "human_label", "human_comment",
        ])
        for item in validity["per_pred_triple"]:
            pred = item["pred_triple"]
            support_strs = []
            for idx in item["supporting_gt_indices"]:
                if idx in gt_triples:
                    support_strs.append(format_triple(gt_triples[idx]))
            writer.writerow([
                item["pred_index"],
                pred["head"],
                pred["relation"],
                pred["tail"],
                "是" if item["valid"] else "否",
                "\n".join(support_strs) if support_strs else "",
                "",
                "",
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_dir",
        default="./results/example/debug_eval_traj_eval",
    )
    parser.add_argument(
        "--output_dir",
        default="./results/example/debug_eval_traj_eval_meta_eval",
    )
    args = parser.parse_args()

    gt_out_dir = os.path.join(args.output_dir, "gt_recall_check")
    pred_out_dir = os.path.join(args.output_dir, "pred_precision_check")
    os.makedirs(gt_out_dir, exist_ok=True)
    os.makedirs(pred_out_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(args.eval_dir, "task_*.json")))
    print(f"Found {len(json_files)} task files")

    for jf in json_files:
        process_one_file(jf, gt_out_dir, pred_out_dir)

    print(f"Done. GT CSVs -> {gt_out_dir}")
    print(f"Done. Pred CSVs -> {pred_out_dir}")


if __name__ == "__main__":
    main()
