#!/usr/bin/env python3
"""Benchmark ways of running the subject-segmentation model.

    python3 tools/bench_segmentation.py                 # every backend
    python3 tools/bench_segmentation.py -b onnx-cpu-768 torch-mps

Each backend runs in its own process with a time limit, so one that stalls
(Core ML compiling a model it can only partly run) cannot hang the
benchmark.  For each it reports load time, median time per photo, peak
memory, and how well its masks agree with the reference - ONNX on the CPU
at the model's native 1024 px, which is what the app ships - as mean IoU.
A faster backend is only useful if that stays close to 1.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.expanduser("~/.u2net/birefnet-general-lite.onnx")
HF_REPO = "ZhengPeng7/BiRefNet_lite"

BACKENDS = {
    # name: (kind, input size, extra)
    "onnx-cpu-1024": ("onnx", 1024, {"providers": ["CPUExecutionProvider"]}),
    "onnx-cpu-768": ("onnx", 768, {"providers": ["CPUExecutionProvider"]}),
    "onnx-cpu-512": ("onnx", 512, {"providers": ["CPUExecutionProvider"]}),
    "onnx-coreml-mlprogram": ("onnx", 1024, {"providers": [
        ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
        "CPUExecutionProvider"]}),
    "torch-mps": ("torch", 1024, {"device": "mps", "half": False}),
    "torch-mps-fp16": ("torch", 1024, {"device": "mps", "half": True}),
    "torch-mps-768": ("torch", 768, {"device": "mps", "half": False}),
}

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _preprocess(rgb, size):
    import cv2
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None]


def _worker(name, photos, out_dir):
    """Runs inside the child process; writes masks and timings."""
    import resource
    import cv2
    kind, size, extra = BACKENDS[name]
    t0 = time.time()
    if kind == "onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(ONNX, providers=extra["providers"])
        inp = sess.get_inputs()[0].name
        print(json.dumps({"stage": "loaded", "s": time.time() - t0,
                          "providers": sess.get_providers()}), flush=True)

        def run(x):
            return sess.run(None, {inp: x})[0][0, 0]
    else:
        import torch
        from transformers import AutoModelForImageSegmentation
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # deform_conv2d
        net = AutoModelForImageSegmentation.from_pretrained(HF_REPO, trust_remote_code=True)
        dev = torch.device(extra["device"])
        net = net.to(dev).eval()
        if extra["half"]:
            net = net.half()
        print(json.dumps({"stage": "loaded", "s": time.time() - t0}), flush=True)

        def run(x):
            t = torch.from_numpy(x).to(dev)
            if extra["half"]:
                t = t.half()
            with torch.no_grad():
                y = net(t)[-1].sigmoid().float()
            if dev.type == "mps":
                torch.mps.synchronize()
            return y[0, 0].cpu().numpy()

    times = []
    for i, p in enumerate(photos):
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        x = _preprocess(rgb, size)
        t = time.time()
        y = run(x)
        times.append(time.time() - t)
        if kind == "onnx":                       # the ONNX export ends in logits
            y = 1.0 / (1.0 + np.exp(-y))
        y = cv2.resize(y.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
        np.save(os.path.join(out_dir, f"{i}.npy"), (y > 0.5).astype(np.uint8))
        print(json.dumps({"stage": "photo", "i": i, "s": times[-1]}), flush=True)
    print(json.dumps({"stage": "done", "first_s": times[0],
                      "median_s": float(np.median(times[1:] or times)),
                      "peak_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9}),
          flush=True)


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else inter / union


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-b", "--backends", nargs="+", default=list(BACKENDS))
    ap.add_argument("-p", "--photos", nargs="+",
                    default=sorted(glob.glob(os.path.join(ROOT, "samples/photos/*.jpg"))))
    ap.add_argument("--timeout", type=float, default=240.0,
                    help="seconds a backend may take in total before it is killed")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        _worker(args.worker, args.photos, args.out)
        return

    work = tempfile.mkdtemp(prefix="segbench-")
    names = ["onnx-cpu-1024"] + [b for b in args.backends if b != "onnx-cpu-1024"]
    results = {}
    for name in names:
        out = os.path.join(work, name)
        os.makedirs(out)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", name, "--out", out,
               "-p", *args.photos]
        print(f"{name:24s} ...", end="", flush=True)
        lines, status = [], "ok"
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout,
                                  env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
            lines = [json.loads(l) for l in proc.stdout.splitlines() if l.startswith("{")]
            if proc.returncode != 0:
                status = "failed: " + (proc.stderr.strip().splitlines() or ["?"])[-1][:120]
        except subprocess.TimeoutExpired as e:
            out_text = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            lines = [json.loads(l) for l in out_text.splitlines() if l.startswith("{")]
            status = f"killed after {args.timeout:.0f} s"
        stages = {l["stage"]: l for l in lines}
        r = {"status": status, **stages.get("done", {})}
        r["load_s"] = stages.get("loaded", {}).get("s")
        r["providers"] = stages.get("loaded", {}).get("providers")
        if "done" in stages:
            ref = os.path.join(work, "onnx-cpu-1024")
            ious = [_iou(np.load(os.path.join(out, f"{i}.npy")),
                         np.load(os.path.join(ref, f"{i}.npy")))
                    for i in range(len(args.photos))]
            r["iou"] = float(np.mean(ious))
            r["iou_min"] = float(np.min(ious))
        results[name] = r
        if "done" in stages:
            print(f"\r{name:24s} load {r['load_s']:6.1f}s  per photo {r['median_s']:6.2f}s  "
                  f"peak {r['peak_gb']:4.1f} GB  IoU {r['iou']:.3f} (worst {r['iou_min']:.3f})")
        else:
            loaded = f"(loaded in {r['load_s']:.1f}s)" if r["load_s"] else "(never finished loading)"
            print(f"\r{name:24s} {status} {loaded}")
    with open(os.path.join(work, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nmasks and results.json in {work}")


if __name__ == "__main__":
    main()
