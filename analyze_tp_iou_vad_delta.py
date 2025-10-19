import os
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import librosa
except Exception as e:
    librosa = None


def parse_window_size(dirname: str) -> float:
    name = os.path.basename(dirname)
    if name.startswith("window_") and name.endswith("s"):
        return float(name[len("window_"):-1])
    raise ValueError(f"Unexpected window dir: {dirname}")


def interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(eval_csv_path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    gt_index: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    with open(eval_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_index[row["file_path"]].append((float(row["start_time"]), float(row["end_time"]), row["label"]))
    for fp in gt_index:
        gt_index[fp].sort(key=lambda x: x[0])
    return gt_index


def find_best_gt_match(gt_list: List[Tuple[float, float, str]], start: float, end: float, label: str) -> Optional[Tuple[float, float, str, float]]:
    best = None
    best_iou = 0.0
    for (gs, ge, glabel) in gt_list:
        if glabel != label:
            continue
        iou = interval_iou(start, end, gs, ge)
        if iou > best_iou:
            best_iou = iou
            best = (gs, ge, glabel, iou)
            if iou >= 1.0:
                break
    return best


def refine_with_vad(file_path: str, start: float, end: float, sample_rate: int = 16000, top_db: int = 20) -> Optional[Tuple[float, float]]:
    if librosa is None:
        return None
    try:
        padding = 0.1
        padded_start = max(0.0, start - padding)
        padded_duration = (end - padded_start) + padding
        audio, sr = librosa.load(file_path, sr=sample_rate, offset=padded_start, duration=padded_duration)
        if len(audio) == 0:
            return None
        speech_intervals = librosa.effects.split(audio, top_db=top_db)
        if len(speech_intervals) == 0:
            return None
        # Collect segments within original span
        segs = []
        for s, e in speech_intervals:
            abs_s = padded_start + (s / sample_rate)
            abs_e = padded_start + (e / sample_rate)
            if abs_e > start and abs_s < end:
                segs.append((max(abs_s, start), min(abs_e, end)))
        if not segs:
            return None
        segs.sort()
        # Merge close segments (<=50ms)
        merged = []
        cs, ce = segs[0]
        for s, e in segs[1:]:
            if s <= ce + 0.05:
                ce = max(ce, e)
            else:
                merged.append((cs, ce))
                cs, ce = s, e
        merged.append((cs, ce))
        # Choose the segment with maximum overlap length within original span
        best_seg = max(merged, key=lambda seg: seg[1] - seg[0])
        return best_seg
    except Exception:
        return None


def collect_merged_prediction_csvs(base_dir: str) -> Dict[float, List[str]]:
    out: Dict[float, List[str]] = defaultdict(list)
    for root, dirs, files in os.walk(base_dir):
        if "merged_predictions.csv" in files:
            parts = root.split(os.sep)
            wins = [p for p in parts if p.startswith("window_") and p.endswith("s")]
            if not wins:
                continue
            try:
                w = parse_window_size(wins[-1])
            except ValueError:
                continue
            out[w].append(os.path.join(root, "merged_predictions.csv"))
    return out


def main():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    gt_csv = os.path.join(repo_root, "csv", "eval.csv")
    window_only_base = os.path.join(repo_root, "fixed_smart_parallel_results", "eval_by_0.05", "eval_by_0.05", "word_eval")

    if not os.path.exists(gt_csv):
        raise FileNotFoundError(gt_csv)
    if not os.path.isdir(window_only_base):
        raise FileNotFoundError(window_only_base)

    gt_index = load_ground_truth(gt_csv)
    mapping = collect_merged_prediction_csvs(window_only_base)

    results = []
    per_window_deltas: Dict[float, List[float]] = defaultdict(list)
    per_window_orig_ious: Dict[float, List[float]] = defaultdict(list)
    per_window_vad_ious: Dict[float, List[float]] = defaultdict(list)

    for w_size, paths in sorted(mapping.items()):
        for csv_path in paths:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_path = row["file_path"]
                    pred_label = row.get("predicted_label")
                    true_label = row.get("true_label")
                    if pred_label is None or true_label is None:
                        continue
                    # True Positive: predicted_label equals true_label and not 'none'
                    if pred_label == "none" or pred_label != true_label:
                        continue
                    start = float(row["start_time"])  # merged span
                    end = float(row["end_time"])      # merged span

                    gt_list = gt_index.get(file_path, [])
                    if not gt_list:
                        continue
                    best = find_best_gt_match(gt_list, start, end, pred_label)
                    if best is None:
                        continue
                    gs, ge, glabel, base_iou = best

                    # Apply VAD on this TP span
                    vad_span = refine_with_vad(file_path, start, end)
                    if vad_span is None:
                        # If VAD fails, keep base IoU as VAD IoU (no change)
                        vad_iou = base_iou
                        vad_start, vad_end = start, end
                    else:
                        vad_start, vad_end = vad_span
                        vad_iou = interval_iou(vad_start, vad_end, gs, ge)

                    delta = (vad_iou - base_iou) * 100.0
                    per_window_deltas[w_size].append(delta)
                    per_window_orig_ious[w_size].append(base_iou)
                    per_window_vad_ious[w_size].append(vad_iou)
                    results.append({
                        "window_size_s": w_size,
                        "file_path": file_path,
                        "label": pred_label,
                        "orig_start": start,
                        "orig_end": end,
                        "gt_start": gs,
                        "gt_end": ge,
                        "orig_iou": base_iou,
                        "vad_start": vad_start,
                        "vad_end": vad_end,
                        "vad_iou": vad_iou,
                        "delta_pp": delta,
                    })

    out_dir = os.path.join(repo_root, "analysis_outputs")
    os.makedirs(out_dir, exist_ok=True)
    detailed_csv = os.path.join(out_dir, "tp_iou_vad_delta_detailed.csv")
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "window_size_s","file_path","label","orig_start","orig_end","gt_start","gt_end","orig_iou","vad_start","vad_end","vad_iou","delta_pp"
        ])
        writer.writeheader()
        writer.writerows(results)

    summary_csv = os.path.join(out_dir, "tp_iou_vad_delta_by_window.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_size_s","tp_count","mean_delta_pp","median_delta_pp","p90_delta_pp"])
        writer.writeheader()
        for w, deltas in sorted(per_window_deltas.items()):
            if deltas:
                writer.writerow({
                    "window_size_s": w,
                    "tp_count": len(deltas),
                    "mean_delta_pp": float(np.mean(deltas)),
                    "median_delta_pp": float(np.median(deltas)),
                    "p90_delta_pp": float(np.percentile(deltas, 90)),
                })

    print(f"Saved detailed TP deltas to: {detailed_csv}")
    print(f"Saved per-window summary to: {summary_csv}")

    # Save original vs VAD IoU comparison per window
    compare_csv = os.path.join(out_dir, "tp_iou_compare_by_window.csv")
    with open(compare_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_size_s",
                "tp_count",
                "mean_orig_iou",
                "mean_vad_iou",
                "mean_orig_iou_pct",
                "mean_vad_iou_pct",
            ],
        )
        writer.writeheader()
        for w in sorted(set(per_window_orig_ious.keys()) | set(per_window_vad_ious.keys())):
            orig_list = per_window_orig_ious.get(w, [])
            vad_list = per_window_vad_ious.get(w, [])
            tp_count = min(len(orig_list), len(vad_list)) if (orig_list and vad_list) else max(len(orig_list), len(vad_list))
            mean_orig = float(np.mean(orig_list)) if orig_list else 0.0
            mean_vad = float(np.mean(vad_list)) if vad_list else 0.0
            writer.writerow({
                "window_size_s": w,
                "tp_count": tp_count,
                "mean_orig_iou": mean_orig,
                "mean_vad_iou": mean_vad,
                "mean_orig_iou_pct": mean_orig * 100.0,
                "mean_vad_iou_pct": mean_vad * 100.0,
            })

    print(f"Saved TP IoU comparison summary to: {compare_csv}")

    # Optional plots
    try:
        import matplotlib.pyplot as plt  # type: ignore
        # Delta plot
        xs = []
        ys = []
        for w, deltas in sorted(per_window_deltas.items()):
            xs.append(str(w))
            ys.append(float(np.mean(deltas)) if deltas else 0.0)
        if xs:
            plt.figure(figsize=(10,4))
            plt.bar(xs, ys, color="#1F77B4")
            plt.axhline(0, color="gray", linewidth=0.8)
            plt.ylabel("Mean IoU delta (pp)")
            plt.xlabel("Window size (s)")
            plt.title("True Positives: IoU improvement after VAD (by window)")
            plt.tight_layout()
            fig_path = os.path.join(out_dir, "tp_iou_vad_delta_by_window.png")
            plt.savefig(fig_path, dpi=200)
            print(f"Saved plot: {fig_path}")

        # Side-by-side comparison plot (original vs VAD)
        win_sizes = sorted(set(per_window_orig_ious.keys()) | set(per_window_vad_ious.keys()))
        if win_sizes:
            x_labels = [str(w) for w in win_sizes]
            orig_means = [float(np.mean(per_window_orig_ious[w]))*100.0 if per_window_orig_ious.get(w) else 0.0 for w in win_sizes]
            vad_means = [float(np.mean(per_window_vad_ious[w]))*100.0 if per_window_vad_ious.get(w) else 0.0 for w in win_sizes]
            x = list(range(len(win_sizes)))
            width = 0.35
            plt.figure(figsize=(12,4))
            plt.bar([i - width/2 for i in x], orig_means, width=width, label="Original (TP)", color="#7F7F7F")
            plt.bar([i + width/2 for i in x], vad_means, width=width, label="VAD (TP)", color="#1F77B4")
            plt.ylabel("Mean IoU (%, TP)")
            plt.xlabel("Window size (s)")
            plt.title("True Positives: Mean IoU before/after VAD by window")
            plt.xticks(x, x_labels)
            plt.legend()
            plt.tight_layout()
            fig2 = os.path.join(out_dir, "tp_iou_compare_by_window.png")
            plt.savefig(fig2, dpi=200)
            print(f"Saved plot: {fig2}")
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    main()
