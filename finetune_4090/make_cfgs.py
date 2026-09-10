"""Generate per-task config JSONs with your local media/data paths.

Usage (from the repo root):
    python finetune_4090/make_cfgs.py --task t1 --data-root D:/datasets/didemo \
        --pretrain-dir D:/weights/gram/GRAM_pretrained_4modalities \
        --out-root D:/runs

It writes:
    finetune_4090/cfgs/{task}_{dataset}.json
The file uses the same schema as config/gram/finetune_cfg/retrieval-*.json so the
native `run.py` / `finetune_4090/run_tasks.py` can consume it.
"""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_CFG = os.path.join(ROOT, "config/gram/default_model_cfg.json")
DEFAULT_RUN_CFG = os.path.join(ROOT, "config/gram/default_run_cfg.json")
ANN_DIR = os.path.join(ROOT, "datasets/annotations")
OUT_CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfgs")


DATASETS = {
    "didemo": {
        "train_ann": "didemo/descs_ret_train.json",
        "val_ann": "didemo/descs_ret_test.json",
        "train_task": "ret%tva",
        "val_task": "ret%tva",
        "default_epochs": 5,
    },
    "msrvtt": {
        "train_ann": "msrvtt/descs_ret_train.json",
        "val_ann": "msrvtt/descs_ret_test.json",
        "train_task": "ret%tv%ta",
        "val_task": "ret%tvas",
        "default_epochs": 3,
    },
    "vatex": {
        "train_ann": "vatex/descs_ret_train.json",
        "val_ann": "vatex/descs_ret_test.json",
        "train_task": "ret%tv%ta",
        "val_task": "ret%tv%ta",
        "default_epochs": 1,
    },
}


def build_json(task, dataset, data_root, epochs, vision_sample_num,
               audio_sample_num, train_bs, val_bs, max_caption_len,
               with_subtitles):
    d = DATASETS[dataset]
    video_dir = os.path.join(data_root, "videos")
    audio_dir = os.path.join(data_root, "audios")
    train_ann = os.path.join(data_root, "annos_ret_train.json")
    val_ann = os.path.join(data_root, "annos_ret_test.json")

    # Local, filtered annotations written by prepare_data.py (falls back to the
    # repo copies, which contain /mnt paths only in the *media* fields -- the
    # actual text is usable, but loader filtering needs the local ann file).
    if not os.path.exists(train_ann):
        train_ann = os.path.join(ANN_DIR, d["train_ann"])
    if not os.path.exists(val_ann):
        val_ann = os.path.join(ANN_DIR, d["val_ann"])

    def _posix(p):
        return p.replace("\\", "/")

    video_dir, audio_dir, train_ann, val_ann = map(
        _posix, (video_dir, audio_dir, train_ann, val_ann))

    val_task = d["val_task"]
    train_task = d["train_task"]
    if dataset == "didemo":
        val_task = "ret%tva"

    common = {
        "type": "annoindexed",
        "vision": video_dir,
        "audio": audio_dir,
        "vision_format": "video_rawvideo",
        "vision_sample_num": vision_sample_num,
        "audio_sample_num": audio_sample_num,
        "n_workers": 4,
    }
    train = dict(common)
    train.update({
        "training": True,
        "name": f"{dataset}_ret",
        "txt": train_ann,
        "task": train_task,
        "epoch": epochs,
        "batch_size": train_bs,
    })
    val = dict(common)
    val.update({
        "training": False,
        "name": f"{dataset}_ret",
        "txt": val_ann,
        "task": val_task,
        "epoch": 1,
        "batch_size": val_bs,
    })

    model_extra = {
        "frozen_vision": True,
        "frozen_audio": True,
        "checkpointing": True,
        "max_caption_len": max_caption_len,
        "itm_ratio": 0.0,           # T1/T2: skip the ITM cross-attention loss
        "itm_rerank_num": 10,       # cap eval-time BERT rerank cost on 4090
        "ret_bidirection_evaluation": False,
    }
    if task == "t3":
        model_extra["checkpointing"] = True

    cfg = {
        "run_cfg": {
            "default": DEFAULT_RUN_CFG,
            "learning_rate": 3e-5 if task == "t1" else 1e-4,
            # NOTE: in this repo `batch_size` is already the per-optimizer-step
            # micro-batch (build_dataloader divides it by world size only and the
            # trainer steps every iteration). Keep accum=1.
            "gradient_accumulation_steps": 1,
            "checkpointing": True,
            "fp16": True,
            "use_ddp": False,
            "mode": "training",
            "output_dir": "none",
            "num_train_steps": 0,
            "first_eval": False,
            "valid_freq": 10,
        },
        "model_cfg": {
            "default": DEFAULT_MODEL_CFG,
            **model_extra,
        },
        "data_cfg": {
            "train": [train],
            "val": [val],
        },
    }
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["t1", "t2", "t3"])
    ap.add_argument("--dataset", default=None,
                    choices=list(DATASETS.keys()),
                    help="default: t1->didemo, t2->msrvtt, t3->vatex")
    ap.add_argument("--data-root", required=True,
                    help="directory containing videos/ audios/ annos_ret_*.json")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--vision-sample-num", type=int, default=4)
    ap.add_argument("--audio-sample-num", type=int, default=1)
    ap.add_argument("--train-bs", type=int, default=2)
    ap.add_argument("--val-bs", type=int, default=1)
    ap.add_argument("--max-caption-len", type=int, default=40)
    args = ap.parse_args()

    if args.dataset is None:
        args.dataset = {"t1": "didemo", "t2": "msrvtt", "t3": "vatex"}[args.task]
    if args.epochs is None:
        args.epochs = DATASETS[args.dataset]["default_epochs"]

    os.makedirs(OUT_CFG_DIR, exist_ok=True)
    cfg = build_json(
        task=args.task,
        dataset=args.dataset,
        data_root=os.path.abspath(args.data_root),
        epochs=args.epochs,
        vision_sample_num=args.vision_sample_num,
        audio_sample_num=args.audio_sample_num,
        train_bs=args.train_bs,
        val_bs=args.val_bs,
        max_caption_len=args.max_caption_len,
        with_subtitles=False,
    )
    out = os.path.join(OUT_CFG_DIR, f"{args.task}_{args.dataset}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("wrote", out)
    if os.path.exists(os.path.join(os.path.abspath(args.data_root), "videos")) and \
            not os.path.exists(os.path.join(os.path.abspath(args.data_root),
                                            "annos_ret_train.json")):
        print("hint: run finetune_4090/prepare_data.py first so annos_ret_*.json "
              "are written next to videos/")
    print("train txt :", cfg["data_cfg"]["train"][0]["txt"])
    print("val txt   :", cfg["data_cfg"]["val"][0]["txt"])
    print("vision    :", cfg["data_cfg"]["train"][0]["vision"])
    print("audio     :", cfg["data_cfg"]["train"][0]["audio"])


if __name__ == "__main__":
    main()
