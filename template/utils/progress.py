"""Low-overhead progress tracker for training pipelines.

Prints stage boundaries / heartbeats and optionally writes every event
as a JSONL line for downstream consumption.
"""
from __future__ import annotations

import json
import time


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m{s:.0f}s"


class ProgressTracker:
    """Low-overhead progress tracker — prints stage boundaries / heartbeats
    and optionally writes every event as a JSONL line."""

    def __init__(self, cfg):
        prog = getattr(cfg, "progress", None)
        self.enabled = prog.enabled if prog else True
        self.every_n_batches = prog.every_n_batches if prog else 10
        self.every_n_steps = prog.every_n_steps if prog else 50
        self.min_interval = prog.min_interval_sec if prog else 10.0
        self._jsonl_path = prog.jsonl_path if prog else ""
        self._jsonl_fh = None
        self._stage_t0: dict[str, float] = {}

    # ── core emit ──

    def _emit(self, msg: str, record: dict | None = None):
        if not self.enabled:
            return
        print(msg, flush=True)
        if self._jsonl_path and record is not None:
            if self._jsonl_fh is None:
                self._jsonl_fh = open(self._jsonl_path, "a", encoding="utf-8")
            record.setdefault("ts", time.time())
            self._jsonl_fh.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )
            self._jsonl_fh.flush()

    # ── stage boundary helpers ──

    def stage_start(self, name: str):
        self._stage_t0[name] = time.time()
        self._emit(f"[Progress] {name} ...", {"event": "stage_start", "stage": name})

    def stage_end(self, name: str, **extra):
        elapsed = time.time() - self._stage_t0.pop(name, time.time())
        parts = [f"[Progress] {name} done in {_fmt_elapsed(elapsed)}"]
        if extra:
            details = ", ".join(f"{k}={v}" for k, v in extra.items())
            parts.append(f"({details})")
        self._emit(
            " ".join(parts),
            {"event": "stage_end", "stage": name,
             "elapsed_sec": round(elapsed, 1), **extra},
        )

    # ── streaming callback factory ──

    def make_streaming_callback(self, stage: str):
        """Return a lightweight (batches_done, rows_seen) callback, or None."""
        if not self.enabled:
            return None
        last_t = [time.time()]
        stage_t0 = self._stage_t0
        every_n = self.every_n_batches
        min_iv = self.min_interval
        emit = self._emit

        def _cb(batches_done: int, rows_seen: int):
            now = time.time()
            if batches_done % every_n == 0 and (now - last_t[0]) >= min_iv:
                elapsed = now - stage_t0.get(stage, now)
                emit(
                    f"[Progress]   {stage}: batch {batches_done}, "
                    f"rows {rows_seen:,}, elapsed {_fmt_elapsed(elapsed)}",
                    {"event": "heartbeat", "stage": stage,
                     "batches_done": batches_done, "rows_seen": rows_seen,
                     "elapsed_sec": round(elapsed, 1)},
                )
                last_t[0] = now
        return _cb

    # ── training step callback factory (per-epoch) ──

    def make_train_step_callback(self, epoch: int, total_epochs: int):
        """Return a (step, total_steps, avg_loss) callback for one epoch."""
        if not self.enabled:
            return None
        last_t = [time.time()]
        epoch_t0 = [time.time()]
        every_n = self.every_n_steps
        min_iv = self.min_interval
        emit = self._emit

        def _cb(step: int, total_steps: int, avg_loss: float):
            now = time.time()
            if step % every_n == 0 and (now - last_t[0]) >= min_iv:
                elapsed = now - epoch_t0[0]
                eta = elapsed / step * (total_steps - step) if step > 0 else 0
                emit(
                    f"  [Epoch {epoch:02d}/{total_epochs}] "
                    f"step {step}/{total_steps}, avg_loss={avg_loss:.6f}, "
                    f"elapsed {_fmt_elapsed(elapsed)}, eta {_fmt_elapsed(eta)}",
                    {"event": "train_step", "epoch": epoch, "step": step,
                     "total_steps": total_steps, "avg_loss": round(avg_loss, 6),
                     "elapsed_sec": round(elapsed, 1), "eta_sec": round(eta, 1)},
                )
                last_t[0] = now
        return _cb

    def close(self):
        if self._jsonl_fh:
            self._jsonl_fh.close()
            self._jsonl_fh = None
