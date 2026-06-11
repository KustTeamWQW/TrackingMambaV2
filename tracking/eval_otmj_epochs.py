import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch

TRACKING_DIR = Path(__file__).resolve().parent
PROJECT_DIR = TRACKING_DIR.parent
for path in (str(TRACKING_DIR), str(PROJECT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import _init_paths  # noqa: F401,E402
from lib.test.analysis.extract_results import extract_results
from lib.test.analysis.plot_results import get_auc_curve, get_prec_curve
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.running import _save_tracker_output
from lib.test.evaluation.tracker import Tracker


def parse_epochs(spec):
    epochs = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start, end = int(start), int(end)
            step = 1 if start <= end else -1
            epochs.extend(range(start, end + step, step))
        else:
            epochs.append(int(part))
    seen = set()
    ordered = []
    for epoch in epochs:
        if epoch not in seen:
            ordered.append(epoch)
            seen.add(epoch)
    return ordered


def result_files_for_epoch(epoch, dataset):
    env = env_settings()
    save_name = f"ep{epoch:04d}"
    result_dir = Path(env.results_path) / "trackingmambav2" / "TrackingmambaV2-ep150-full-256" / "otmj" / save_name
    return result_dir, [result_dir / f"{seq.name}.txt" for seq in dataset]


def run_epoch(epoch, threads, num_gpus, force):
    dataset = get_dataset("otmj")
    result_dir, result_files = result_files_for_epoch(epoch, dataset)
    if not force and result_files and all(path.is_file() for path in result_files):
        print(f"[eval] epoch {epoch:04d}: existing complete results, skip tracking")
        return

    os.environ["TRACKINGMAMBAV2_TEST_EPOCH"] = str(epoch)
    torch.cuda.set_device(0)

    tracker_info = Tracker(
        "trackingmambav2",
        "TrackingmambaV2-ep150-full-256",
        "otmj",
        f"ep{epoch:04d}",
        None,
    )
    params = tracker_info.get_parameters()
    params.debug = 0
    tracker = tracker_info.create_tracker(params)

    print(f"[eval] epoch {epoch:04d}: running OTMJ with one model load on {len(dataset)} sequences")
    for seq in dataset:
        bbox_file = result_dir / f"{seq.name}.txt"
        if bbox_file.is_file() and not force:
            print(f"[eval] epoch {epoch:04d}: skip existing sequence {seq.name}")
            continue

        print(f"Tracker: trackingmambav2 TrackingmambaV2-ep150-full-256 None ,  Sequence: {seq.name}")
        try:
            output = tracker_info._track_sequence(tracker, seq, seq.init_info())
        except Exception as exc:
            print(f"[eval] epoch {epoch:04d}: sequence {seq.name} failed: {exc}")
            continue

        if isinstance(output["time"][0], dict):
            exec_time = sum(sum(times.values()) for times in output["time"])
            num_frames = len(output["time"])
        else:
            exec_time = sum(output["time"])
            num_frames = len(output["time"])
        print("FPS: {}".format(num_frames / exec_time if exec_time > 0 else 0.0))
        _save_tracker_output(seq, tracker_info, output)

    del tracker
    torch.cuda.empty_cache()
    time.sleep(0.2)


def score_epoch(epoch, force_eval=False):
    dataset = get_dataset("otmj")
    save_name = f"ep{epoch:04d}"
    trackers = trackerlist(
        name="trackingmambav2",
        parameter_name="TrackingmambaV2-ep150-full-256",
        dataset_name="otmj",
        save_name=save_name,
        run_ids=None,
        display_name=f"TrackingmambaV2_ep{epoch:04d}",
    )
    eval_data = extract_results(
        trackers,
        dataset,
        report_name=f"otmj_epoch_sweep/{save_name}",
        skip_missing_seq=False,
        plot_bin_gap=0.05,
        exclude_invalid_frames=False,
    )
    valid_sequence = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    success_curve = torch.tensor(eval_data["ave_success_rate_plot_overlap"])
    center_curve = torch.tensor(eval_data["ave_success_rate_plot_center"])
    norm_center_curve = torch.tensor(eval_data["ave_success_rate_plot_center_norm"])

    _, auc = get_auc_curve(success_curve, valid_sequence)
    _, precision_score = get_prec_curve(center_curve, valid_sequence)
    _, norm_precision_score = get_prec_curve(norm_center_curve, valid_sequence)

    success = auc[0].item()
    precision = precision_score[0].item()
    norm_precision = norm_precision_score[0].item()
    ao = torch.tensor(eval_data["avg_overlap_all"])[valid_sequence, 0].mean().item() * 100.0
    return {
        "epoch": epoch,
        "success": success,
        "precision": precision,
        "norm_precision": norm_precision,
        "ao": ao,
    }


def write_scores(scores, out_file):
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "success", "precision", "norm_precision", "ao"])
        writer.writeheader()
        writer.writerows(scores)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TrackingmambaV2 checkpoints on OTMJ.")
    parser.add_argument("--epochs", default="280-150", help="Comma-separated epochs or ranges, e.g. 280-150,245,260.")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", default="output/test/otmj_epoch_sweep.csv")
    args = parser.parse_args()

    epochs = parse_epochs(args.epochs)
    scores = []
    for epoch in epochs:
        if not args.score_only:
            run_epoch(epoch, args.threads, args.num_gpus, args.force)
        score = score_epoch(epoch)
        scores.append(score)
        write_scores(scores, args.out)
        print(
            "[score] epoch {epoch:04d}: success={success:.3f}, precision={precision:.3f}, "
            "norm_precision={norm_precision:.3f}, ao={ao:.3f}".format(**score)
        )

    best = max(scores, key=lambda row: row["success"])
    print(
        "[best_success] epoch {epoch:04d}: success={success:.3f}, precision={precision:.3f}, "
        "norm_precision={norm_precision:.3f}, ao={ao:.3f}".format(**best)
    )


if __name__ == "__main__":
    main()
