#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


# -----------------------------
# ラベル定義
# -----------------------------
LABEL_MAP = {
    "W": 0,
    "RING": 1,
    "SLOPE": 2,
    "UNK": 3,
    "UNKNOWN": 3,
    "NONE": 3,
}
LABEL_NAMES = ["W", "RING", "SLOPE", "UNK"]


@dataclass
class SampleRaw:
    t_ms: int
    xr: int
    yr: int
    zr: int
    lineno: int


@dataclass
class MarkerRaw:
    t_ms: int
    label: str
    gid: int
    ev: str  # START/END
    lineno: int


@dataclass
class Sample:
    t_ms: int      # epoch補正後の単調増加時刻
    xr: int
    yr: int
    zr: int
    epoch: int
    lineno: int


@dataclass
class Marker:
    t_ms: int      # epoch補正後の単調増加時刻
    label: str
    gid: int
    ev: str
    epoch: int
    lineno: int


@dataclass
class Interval:
    label: str
    gid: int
    t_start: int
    t_end: int
    epoch: int


# -----------------------------
# ログ読取：S/M/H以外は無視（ACK混在対応）
# -----------------------------
def parse_log(path: str) -> Tuple[List[SampleRaw], List[MarkerRaw], Dict[str, str]]:
    samples: List[SampleRaw] = []
    markers: List[MarkerRaw] = []
    headers: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # ACKやコメント行を無視
            if not (line.startswith("S,") or line.startswith("M,") or line.startswith("H,")):
                continue

            parts = [p.strip() for p in line.split(",")]
            tag = parts[0]

            if tag == "H":
                if len(parts) >= 3:
                    headers[parts[1]] = ",".join(parts[2:])
                continue

            if tag == "S" and len(parts) >= 5:
                try:
                    t_ms = int(parts[1])
                    xr = int(parts[2]); yr = int(parts[3]); zr = int(parts[4])
                except ValueError:
                    continue
                samples.append(SampleRaw(t_ms, xr, yr, zr, lineno))
                continue

            if tag == "M" and len(parts) >= 5:
                try:
                    t_ms = int(parts[1])
                    label = parts[2].upper()
                    gid = int(parts[3])
                    ev = parts[4].upper()
                except ValueError:
                    continue
                if ev not in ("START", "END"):
                    continue
                markers.append(MarkerRaw(t_ms, label, gid, ev, lineno))
                continue

    return samples, markers, headers


# -----------------------------
# epoch補正：時刻が減少したらepochを進め、単調増加に変換
#   offset += (prev_t + expected_dt) - cur_t
# -----------------------------
def apply_epoch_correction(samples_raw: List[SampleRaw],
                           markers_raw: List[MarkerRaw],
                           expected_dt_ms: int) -> Tuple[List[Sample], List[Marker], int]:
    # ファイル順（lineno）で統合して走査し、同一ロジックでepochを進める
    events = []
    for s in samples_raw:
        events.append(("S", s.lineno, s))
    for m in markers_raw:
        events.append(("M", m.lineno, m))
    events.sort(key=lambda x: x[1])  # lineno順

    offset = 0
    epoch = 0
    prev_t: Optional[int] = None

    samples: List[Sample] = []
    markers: List[Marker] = []

    for kind, _, obj in events:
        t = obj.t_ms
        if prev_t is not None and t < prev_t:
            # 時刻巻き戻り：新epoch
            offset += (prev_t + expected_dt_ms) - t
            epoch += 1
        t_corr = t + offset

        if kind == "S":
            s = obj
            samples.append(Sample(t_corr, s.xr, s.yr, s.zr, epoch, s.lineno))
        else:
            m = obj
            markers.append(Marker(t_corr, m.label, m.gid, m.ev, epoch, m.lineno))

        prev_t = t

    # 念のため、samples/markersを lineno順に整列（すでにそうなっている）
    samples.sort(key=lambda x: x.lineno)
    markers.sort(key=lambda x: x.lineno)

    return samples, markers, epoch + 1


# -----------------------------
# マーカーを Interval に変換（epoch+gidでペアリング）
# -----------------------------
def markers_to_intervals(markers: List[Marker]) -> List[Interval]:
    start_map: Dict[Tuple[int, int], Marker] = {}  # (epoch,gid) -> START marker
    intervals: List[Interval] = []

    for m in markers:
        key = (m.epoch, m.gid)
        if m.ev == "START":
            start_map[key] = m
        elif m.ev == "END":
            if key in start_map:
                st = start_map.pop(key)
                # END側labelよりSTART側labelを採用
                intervals.append(Interval(st.label, m.gid, st.t_ms, m.t_ms, m.epoch))

    intervals.sort(key=lambda it: (it.epoch, it.t_start))
    return intervals


# -----------------------------
# タイミング統計（補正後）
# -----------------------------
def timing_stats(t_arr: np.ndarray, expected_dt_ms: int) -> Dict[str, int]:
    if len(t_arr) < 2:
        return {"count": int(len(t_arr)), "strict_bad": 0, "min_dt": 0, "max_dt": 0}
    dts = np.diff(t_arr)
    strict_bad = int(np.sum(dts != expected_dt_ms))
    return {
        "count": int(len(t_arr)),
        "strict_bad": strict_bad,
        "min_dt": int(np.min(dts)),
        "max_dt": int(np.max(dts)),
    }


# -----------------------------
# C実装の逐次前処理を再現（epoch/大ギャップでリセット）
# -----------------------------
def preprocess_stream(samples: List[Sample],
                      scale_g_per_lsb: float,
                      alpha: float,
                      gain: float,
                      clip_g: float,
                      reset_on_epoch: bool,
                      reset_on_gap_ms: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    returns:
      t_arr: (N,) int64
      xyz  : (N,3) float32 (clip後)
      epoch: (N,) int32
    """
    n = len(samples)
    t_arr = np.zeros((n,), dtype=np.int64)
    xyz = np.zeros((n, 3), dtype=np.float32)
    ep_arr = np.zeros((n,), dtype=np.int32)

    ax = ay = az = 0.0
    prev_t = None
    prev_ep = None

    for i, s in enumerate(samples):
        t_arr[i] = s.t_ms
        ep_arr[i] = s.epoch

        # リセット条件
        if prev_t is not None:
            gap = s.t_ms - prev_t
        else:
            gap = 0

        if reset_on_epoch and prev_ep is not None and s.epoch != prev_ep:
            ax = ay = az = 0.0
        elif prev_t is not None and gap >= reset_on_gap_ms:
            ax = ay = az = 0.0

        x_g = s.xr * scale_g_per_lsb
        y_g = s.yr * scale_g_per_lsb
        z_g = s.zr * scale_g_per_lsb

        ax = alpha * ax + (1.0 - alpha) * x_g
        ay = alpha * ay + (1.0 - alpha) * y_g
        az = alpha * az + (1.0 - alpha) * z_g

        x = (x_g - ax) * gain
        y = (y_g - ay) * gain
        z = (z_g - az) * gain

        # clip
        if x > clip_g: x = clip_g
        if x < -clip_g: x = -clip_g
        if y > clip_g: y = clip_g
        if y < -clip_g: y = -clip_g
        if z > clip_g: z = clip_g
        if z < -clip_g: z = -clip_g

        xyz[i, 0] = x
        xyz[i, 1] = y
        xyz[i, 2] = z

        prev_t = s.t_ms
        prev_ep = s.epoch

    return t_arr, xyz, ep_arr


# -----------------------------
# 窓生成（dtギャップを跨ぐ窓は除外）
# -----------------------------
def build_windows(t_arr: np.ndarray,
                  xyz: np.ndarray,
                  ep_arr: np.ndarray,
                  intervals: List[Interval],
                  expected_dt_ms: int,
                  max_dt_ms: int,
                  window: int,
                  stride: int,
                  pre_ms: int,
                  post_ms: int,
                  zscore_eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, int, int, int, int, int]]]:
    """
    returns:
      X: (N,window,3)
      y: (N,)
      meta: (label, gid, epoch, t_start, t_end, windows_generated)
    """
    # dtが大きい箇所を検出（窓内に含むなら除外）
    if len(t_arr) >= 2:
        dts = np.diff(t_arr)
        bad_gap = (dts > max_dt_ms) | (dts <= 0)  # 0以下も不正として扱う
        bad_prefix = np.zeros((len(t_arr),), dtype=np.int32)  # bad_prefix[i] = bad数 in dts[0:i]
        bad_prefix[1:] = np.cumsum(bad_gap.astype(np.int32))
    else:
        bad_prefix = np.zeros((len(t_arr),), dtype=np.int32)

    def window_has_gap(a: int, b: int) -> bool:
        # window indices [a, b) -> dt indices [a, b-1)
        if b - a <= 1:
            return False
        return (bad_prefix[b-1] - bad_prefix[a]) > 0

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    meta_list: List[Tuple[str, int, int, int, int, int]] = []

    for it in intervals:
        lab = it.label.upper()
        if lab not in LABEL_MAP:
            meta_list.append((lab, it.gid, it.epoch, it.t_start, it.t_end, 0))
            continue

        y = LABEL_MAP[lab]
        seg0 = it.t_start - pre_ms
        seg1 = it.t_end + post_ms

        # epochが一致する範囲で抽出（epochまたぎは除外）
        # まず epoch一致のサンプル範囲を絞るために、epochでマスクして検索
        idx_ep = np.where(ep_arr == it.epoch)[0]
        if idx_ep.size == 0:
            meta_list.append((lab, it.gid, it.epoch, it.t_start, it.t_end, 0))
            continue

        # epoch内の時刻配列
        t_ep = t_arr[idx_ep]

        i0_rel = int(np.searchsorted(t_ep, seg0, side="left"))
        i1_rel = int(np.searchsorted(t_ep, seg1, side="right"))

        if i1_rel - i0_rel < window:
            meta_list.append((lab, it.gid, it.epoch, it.t_start, it.t_end, 0))
            continue

        # グローバルindexへ戻す
        i0 = int(idx_ep[i0_rel])
        i1 = int(idx_ep[i1_rel - 1]) + 1  # [i0, i1)

        count_w = 0
        for a in range(i0, i1 - window + 1, stride):
            b = a + window

            # epoch一定（安全のため再チェック）
            if ep_arr[a] != ep_arr[b-1]:
                continue

            # 大きなギャップを跨ぐ窓は除外
            if window_has_gap(a, b):
                continue

            w = xyz[a:b].astype(np.float32, copy=True)  # (window,3)

            # 窓全体Z-score（全要素でmean/var：C実装一致）
            flat = w.reshape(-1)
            mean = float(flat.mean())
            var = float(((flat - mean) ** 2).mean())
            inv_std = 1.0 / np.sqrt(var + zscore_eps)
            w = (w - mean) * inv_std

            X_list.append(w)
            y_list.append(y)
            count_w += 1

        meta_list.append((lab, it.gid, it.epoch, it.t_start, it.t_end, count_w))

    if not X_list:
        return np.empty((0, window, 3), dtype=np.float32), np.empty((0,), dtype=np.int64), meta_list

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, meta_list


def main():
    ap = argparse.ArgumentParser(description="Build dataset (.npz) from ESP32 ADXL345 S/M mixed log (ms) with possible time discontinuities.")
    ap.add_argument("--in", dest="inp", required=True, help="input log file (csv/text): contains S/M/H + ACK lines")
    ap.add_argument("--out", default="dataset.npz", help="output npz")

    # 収集・窓パラメータ
    ap.add_argument("--fs", type=int, default=100)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--pre_ms", type=int, default=200)
    ap.add_argument("--post_ms", type=int, default=200)

    # 前処理定数（C実装一致）
    ap.add_argument("--scale", type=float, default=0.004, help="g/LSB")
    ap.add_argument("--alpha", type=float, default=0.90)
    ap.add_argument("--gain", type=float, default=1.5)
    ap.add_argument("--clip", type=float, default=2.0)

    # 途切れ対策
    ap.add_argument("--max_dt_ms", type=int, default=20, help="window内で許容する最大dt(ms)。これを超えるdtを跨ぐ窓は破棄")
    ap.add_argument("--reset_on_epoch", action="store_true", help="epoch境界でDC除去状態をリセット（推奨）")
    ap.add_argument("--reset_on_gap_ms", type=int, default=200, help="このms以上のギャップでDC除去状態をリセット")

    args = ap.parse_args()

    expected_dt_ms = int(round(1000 / args.fs))

    samples_raw, markers_raw, headers = parse_log(args.inp)
    if len(samples_raw) < args.window:
        raise SystemExit(f"Not enough samples: {len(samples_raw)} (need >= {args.window})")

    samples, markers, epoch_count = apply_epoch_correction(samples_raw, markers_raw, expected_dt_ms)
    intervals = markers_to_intervals(markers)

    # 前処理
    t_arr, xyz, ep_arr = preprocess_stream(
        samples,
        scale_g_per_lsb=args.scale,
        alpha=args.alpha,
        gain=args.gain,
        clip_g=args.clip,
        reset_on_epoch=args.reset_on_epoch,
        reset_on_gap_ms=args.reset_on_gap_ms,
    )

    # 窓生成
    X, y, meta = build_windows(
        t_arr=t_arr,
        xyz=xyz,
        ep_arr=ep_arr,
        intervals=intervals,
        expected_dt_ms=expected_dt_ms,
        max_dt_ms=args.max_dt_ms,
        window=args.window,
        stride=args.stride,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
    )

    # サマリ
    st = timing_stats(t_arr, expected_dt_ms)
    print("[INFO] input:", args.inp)
    print("[INFO] epochs:", epoch_count)
    print("[INFO] samples:", st["count"], " expected_dt_ms=", expected_dt_ms,
          " strict_bad_dt=", st["strict_bad"], " min_dt=", st["min_dt"], " max_dt=", st["max_dt"])
    print("[INFO] markers:", len(markers), " intervals:", len(intervals))
    print("[INFO] windows:", X.shape[0], " X.shape:", X.shape)

    if X.shape[0] > 0:
        uniq, cnt = np.unique(y, return_counts=True)
        for u, c in zip(uniq, cnt):
            name = LABEL_NAMES[int(u)] if int(u) < len(LABEL_NAMES) else str(int(u))
            print(f"  class {int(u)} ({name}): {int(c)} windows")
    else:
        print("[WARN] No windows generated. Check markers / window/stride / max_dt_ms / pre/post.")

    # 保存
    np.savez_compressed(
        args.out,
        X=X,
        y=y,
        label_names=np.array(LABEL_NAMES, dtype=object),
        fs_hz=args.fs,
        window=args.window,
        stride=args.stride,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        scale_g_per_lsb=args.scale,
        alpha=args.alpha,
        gain=args.gain,
        clip_g=args.clip,
        expected_dt_ms=expected_dt_ms,
        max_dt_ms=args.max_dt_ms,
        reset_on_epoch=bool(args.reset_on_epoch),
        reset_on_gap_ms=args.reset_on_gap_ms,
        headers=headers,
        source=args.inp,
    )
    print("[OK] saved:", args.out)


if __name__ == "__main__":
    main()
