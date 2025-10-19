import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple


def parse_window_size(dirname: str) -> float:
    # Expect pattern like 'window_0.5s' -> 0.5
    name = os.path.basename(dirname)
    if name.startswith("window_") and name.endswith("s"):
        try:
            return float(name[len("window_") : -1])
        except ValueError:
            pass
    raise ValueError(f"Unexpected window dir name: {dirname}")


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    # 1D IoU for time intervals [start, end)
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def load_ground_truth(eval_csv_path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    gt_index: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row["file_path"]
            start = float(row["start_time"])
            end = float(row["end_time"])
            label = row["label"]
            gt_index[file_path].append((start, end, label))
    # Sort intervals per file to speed search slightly
    for fp in gt_index:
        gt_index[fp].sort(key=lambda x: x[0])
    return gt_index


def compute_iou_stats_for_preds(pred_csv_path: str, gt_index: Dict[str, List[Tuple[float, float, str]]]) -> Tuple[float, int, int]:
    """
    Returns: (sum_iou_matched, matched_count, total_pred_events_considered)
    We consider only predictions with predicted_label != 'none'.
    A prediction is matched if IoU>0 with a GT interval of the same label; we use the max IoU among same-file same-label GT intervals.
    """
    sum_iou = 0.0
    matched = 0
    total = 0

    with open(pred_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Column names differ slightly between runs; normalize
        # Expected columns include: file_path, start_time, end_time, predicted_label
        for row in reader:
            pred_label = row.get("predicted_label")
            if pred_label is None:
                # Some window-only files may have different header casing
                pred_label = row.get("predicted_class") or row.get("predicted")
            if pred_label is None:
                continue
            if pred_label == "none":
                continue
            try:
                start = float(row["start_time"])  # type: ignore[index]
                end = float(row["end_time"])  # type: ignore[index]
            except Exception:
                # Skip malformed rows
                continue
            file_path = row.get("file_path")
            if not file_path:
                continue
            total += 1

            gt_list = gt_index.get(file_path, [])
            if not gt_list:
                continue
            # Find GT intervals with same label and compute max IoU
            max_iou = 0.0
            for (gs, ge, glabel) in gt_list:
                if glabel != pred_label:
                    continue
                iou = interval_iou(start, end, gs, ge)
                if iou > max_iou:
                    max_iou = iou
                    # Early exit if perfect match
                    if max_iou >= 1.0:
                        break
            if max_iou > 0.0:
                matched += 1
                sum_iou += max_iou

    return sum_iou, matched, total


def collect_pred_paths(base_dir: str) -> Dict[float, List[str]]:
    """Return mapping: window_size -> list of prediction CSV paths under base_dir.
    Looks for 'merged_predictions.csv' under .../window_*s/stride_*/.
    """
    mapping: Dict[float, List[str]] = defaultdict(list)
    for root, dirs, files in os.walk(base_dir):
        if "merged_predictions.csv" in files:
            # Identify window dir ancestor
            parts = root.split(os.sep)
            # Find a component that starts with 'window_' and ends with 's'
            window_dirs = [p for p in parts if p.startswith("window_") and p.endswith("s")]
            if not window_dirs:
                continue
            try:
                w = parse_window_size(window_dirs[-1])
            except ValueError:
                continue
            mapping[w].append(os.path.join(root, "merged_predictions.csv"))
    return mapping


def weighted_mean(sum_vals: float, count: int) -> float:
    return sum_vals / count if count > 0 else 0.0


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))

    # Inputs
    gt_csv = os.path.join(repo_root, "csv", "eval.csv")
    fixed_base = os.path.join(
        repo_root, "fixed_smart_parallel_results", "eval_by_0.05", "eval_by_0.05", "word_eval"
    )
    vad_base = os.path.join(
        repo_root, "vad_refined_results", "vad_refined_results", "eval_by_0.05"
    )

    if not os.path.exists(gt_csv):
        raise FileNotFoundError(f"Ground truth CSV not found: {gt_csv}")
    if not os.path.isdir(fixed_base):
        raise FileNotFoundError(f"Fixed/window-only base dir not found: {fixed_base}")
    if not os.path.isdir(vad_base):
        raise FileNotFoundError(f"VAD-refined base dir not found: {vad_base}")

    print("Loading ground truth...")
    gt_index = load_ground_truth(gt_csv)

    print("Scanning prediction outputs (window-only)...")
    fixed_map = collect_pred_paths(fixed_base)
    print(f"Found window-only window sizes: {sorted(fixed_map.keys())}")
    print("Scanning prediction outputs (VAD-refined)...")
    vad_map = collect_pred_paths(vad_base)
    print(f"Found VAD-refined window sizes: {sorted(vad_map.keys())}")

    common_windows = sorted(set(fixed_map.keys()) & set(vad_map.keys()))
    if not common_windows:
        raise RuntimeError("No common window sizes between window-only and VAD-refined runs.")

    results_rows = []
    summary: Dict[float, Dict[str, float]] = {}

    for w in common_windows:
        # Aggregate across all strides for this window size, weighted by matched count
        fixed_sum_iou = 0.0
        fixed_matched = 0
        fixed_total = 0
        for p in fixed_map[w]:
            s_iou, m_cnt, t_cnt = compute_iou_stats_for_preds(p, gt_index)
            fixed_sum_iou += s_iou
            fixed_matched += m_cnt
            fixed_total += t_cnt

        vad_sum_iou = 0.0
        vad_matched = 0
        vad_total = 0
        for p in vad_map[w]:
            s_iou, m_cnt, t_cnt = compute_iou_stats_for_preds(p, gt_index)
            vad_sum_iou += s_iou
            vad_matched += m_cnt
            vad_total += t_cnt

        fixed_mean = weighted_mean(fixed_sum_iou, fixed_matched)
        vad_mean = weighted_mean(vad_sum_iou, vad_matched)
        delta_pp = (vad_mean - fixed_mean) * 100.0

        summary[w] = {
            "fixed_mean_iou": fixed_mean,
            "fixed_matched": float(fixed_matched),
            "fixed_total_preds": float(fixed_total),
            "vad_mean_iou": vad_mean,
            "vad_matched": float(vad_matched),
            "vad_total_preds": float(vad_total),
            "delta_iou_pp": delta_pp,
        }

        results_rows.append(
            {
                "window_size_s": w,
                "fixed_mean_iou": fixed_mean,
                "fixed_matched": fixed_matched,
                "fixed_total_preds": fixed_total,
                "vad_mean_iou": vad_mean,
                "vad_matched": vad_matched,
                "vad_total_preds": vad_total,
                "delta_iou_percentage_points": delta_pp,
            }
        )

    out_dir = os.path.join(repo_root, "analysis_outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "iou_vad_diff_by_window.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_size_s",
                "fixed_mean_iou",
                "fixed_matched",
                "fixed_total_preds",
                "vad_mean_iou",
                "vad_matched",
                "vad_total_preds",
                "delta_iou_percentage_points",
            ],
        )
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"Saved CSV summary to: {out_csv}")

    # Plot differences using matplotlib if available
    try:
        import matplotlib.pyplot as plt  # type: ignore

        xs = sorted(summary.keys())
        deltas = [summary[w]["delta_iou_pp"] for w in xs]
        fixed_means = [summary[w]["fixed_mean_iou"] * 100.0 for w in xs]
        vad_means = [summary[w]["vad_mean_iou"] * 100.0 for w in xs]

        # Bar plot for delta
        plt.figure(figsize=(10, 4))
        plt.bar([str(x) for x in xs], deltas, color=["#4472C4" if d >= 0 else "#C00000" for d in deltas])
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.ylabel("IoU improvement (pp)")
        plt.xlabel("Window size (s)")
        plt.title("VAD-refined vs Window-only: IoU improvement by window size")
        for i, d in enumerate(deltas):
            plt.text(i, d + (1.0 if d >= 0 else -1.0), f"{d:.1f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=8)
        fig_path = os.path.join(out_dir, "iou_vad_diff_by_window.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        print(f"Saved plot: {fig_path}")

        # Optional: overlay raw IoU means
        plt.figure(figsize=(10, 4))
        xlabels = [str(x) for x in xs]
        x = list(range(len(xs)))
        width = 0.35
        plt.bar([i - width / 2 for i in x], fixed_means, width=width, label="Window-only", color="#7F7F7F")
        plt.bar([i + width / 2 for i in x], vad_means, width=width, label="VAD-refined", color="#1F77B4")
        plt.ylabel("Mean IoU (%, matched)")
        plt.xlabel("Window size (s)")
        plt.title("Mean IoU (matched predictions)")
        plt.xticks(x, xlabels)
        plt.legend()
        fig_path2 = os.path.join(out_dir, "iou_means_by_window.png")
        plt.tight_layout()
        plt.savefig(fig_path2, dpi=200)
        print(f"Saved plot: {fig_path2}")
    except Exception as e:
        print("Matplotlib plotting skipped:", e)
        print("CSV summary is available; install matplotlib to generate plots.")


if __name__ == "__main__":
    main()
