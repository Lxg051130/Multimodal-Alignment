"""Prepare local video+audio layout for a GRAM fine-tuning dataset.

Steps:
1. (optional) extract `<id>.mp3` from each `<id>.mp4` with ffmpeg;
2. intersect the repo annotation JSONs with the files that actually exist
   locally, and write `annos_ret_train.json` / `annos_ret_test.json` into the
   same data root (the paths used by the generated task configs).

Example:
    python finetune_4090/prepare_data.py --name msrvtt \\
        --data-root D:/datasets/msrvtt --video-dir D:/datasets/msrvtt/videos \\
        --extract-audio
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(REPO, "datasets", "annotations")

SPLIT_FILES = {
    "didemo": ("didemo/descs_ret_train.json", "didemo/descs_ret_test.json"),
    "msrvtt": ("msrvtt/descs_ret_train.json", "msrvtt/descs_ret_test.json"),
    "vatex": ("vatex/descs_ret_train.json", "vatex/descs_ret_test.json"),
    "activitynet": ("activitynet/descs_ret_train.json", "activitynet/descs_ret_test.json"),
}


def id_of(anno):
    return anno.get("video_id") or anno.get("id") or anno.get("clip_id")


def video_exists(video_dir, vid, exts=(".mp4", ".avi", ".webm", ".mkv", ".mov")):
    return any(os.path.exists(os.path.join(video_dir, str(vid) + e)) for e in exts)


def extract_audio(video_dir, audio_dir, vid, ffmpeg):
    src = None
    for e in (".mp4", ".avi", ".webm", ".mkv", ".mov"):
        p = os.path.join(video_dir, str(vid) + e)
        if os.path.exists(p):
            src = p
            break
    if src is None:
        return False
    dst = os.path.join(audio_dir, str(vid) + ".mp3")
    if os.path.exists(dst):
        return True
    os.makedirs(audio_dir, exist_ok=True)
    r = subprocess.run(
        [ffmpeg, "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", "-q:a", "4", dst],
        capture_output=True)
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=list(SPLIT_FILES))
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--audio-dir", default="")
    ap.add_argument("--data-root", default="",
                    help="output dir for annos_ret_*.json (default: video-dir parent)")
    ap.add_argument("--extract-audio", action="store_true")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--limit", type=int, default=0,
                    help="dev/debug: keep at most N *unique videos* per split")
    ap.add_argument("--cap-per-video", type=int, default=0,
                    help="dev/debug: keep at most N descriptions per unique "
                         "video (0 = keep all, e.g. 1 = one caption/video)")
    args = ap.parse_args()

    video_dir = os.path.abspath(args.video_dir)
    audio_dir = os.path.abspath(args.audio_dir) if args.audio_dir else ""
    data_root = os.path.abspath(args.data_root or os.path.dirname(video_dir))
    if args.extract_audio:
        assert args.audio_dir, "--extract-audio requires --audio-dir"

    # A single limited video pool shared by both splits, so a small smoke test
    # still has a non-empty test set.
    limited_vids = None
    if args.limit:
        limited_vids = set()
        rels = SPLIT_FILES[args.name]
        budgets = [max(1, int(args.limit * 0.8)), args.limit]
        for rel, budget in zip(rels, budgets):
            annos = json.load(open(os.path.join(ANN, rel), encoding="utf-8"))
            for a in annos:
                vid = id_of(a)
                if vid is not None and video_exists(video_dir, vid):
                    limited_vids.add(vid)
                    if len(limited_vids) >= budget:
                        break
        print(f"[limit] smoke pool: {len(limited_vids)} unique videos")

    for rel in SPLIT_FILES[args.name]:
        src = os.path.join(ANN, rel)
        split_name = "train" if "train" in rel else "test"
        out = os.path.join(data_root, f"annos_ret_{split_name}.json")
        annos = json.load(open(src, encoding="utf-8"))
        keep = []
        seen = set()
        per_video = {}
        for a in annos:
            vid = id_of(a)
            if vid is None:
                continue
            if args.limit and vid not in limited_vids:
                continue
            if vid not in seen and not video_exists(video_dir, vid):
                continue
            seen.add(vid)
            if args.cap_per_video and per_video.get(vid, 0) >= args.cap_per_video:
                continue
            per_video[vid] = per_video.get(vid, 0) + 1
            keep.append(a)
        os.makedirs(data_root, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(keep, f, ensure_ascii=False, indent=1)
        print(f"{split_name}: {len(keep)} rows, {len(per_video)} videos "
              f"-> {out}")
        if args.extract_audio:
            ok = 0
            for a in keep:
                ok += int(extract_audio(video_dir, audio_dir, id_of(a), args.ffmpeg))
            print(f"{split_name}: extracted audio for {ok}/{len(seen)} videos")


if __name__ == "__main__":
    main()
