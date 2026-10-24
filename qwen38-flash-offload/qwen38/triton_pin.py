"""Pin Triton autotune picks of exllamav3's vendored FLA (KDA linear-attention) kernels, so a fresh container
reproduces a reference pick set instead of benchmarking (autotune picks are timing-dependent, and different picks
change the rounding of the KDA prefill: on the teacher-forced panel a different pick set measured top-1 0.989 /
KL 0.0034 vs 1.0000 / 0 for the reference set).

  QWEN38_TRITON_PIN=data/triton_pin_exact.json   (made by make_triton_pin.py from the reference Triton cache)
  format v2: {"<kernel fn name>": {"keys": [[<tuning key list>, <cfg>], ...], "default": <cfg>}}
  format v1: {"<kernel fn name>": <cfg>}   (one config for every key)
  cfg = {"kwargs": {...}, "num_warps": w, "num_stages": s}

v2: every listed key gets its recorded pick (pre-filled into Autotuner.cache, so no benchmark and no disk lookup);
any key NOT listed (e.g. a new chunk size) gets "default" instead of a fresh, timing-dependent benchmark.
install() must run before the first KDA forward (serve.load calls it before model_init.init)."""
import importlib, json, os, pkgutil

PINNED = {}


def tuners():
    import exllamav3.vendor.fla as fla
    from triton.runtime.autotuner import Autotuner
    seen = set()
    for m in pkgutil.walk_packages(fla.__path__, prefix="exllamav3.vendor.fla."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:
            continue
        for v in vars(mod).values():
            x = v
            for _ in range(4):
                if isinstance(x, Autotuner):
                    if id(x) not in seen:
                        seen.add(id(x)); yield x.base_fn.__name__, x
                    break
                x = getattr(x, "fn", None)
                if x is None:
                    break


def _match(t, w):
    m = [c for c in t.configs if c.kwargs == w.get("kwargs", {}) and c.num_warps == w["num_warps"] and c.num_stages == w["num_stages"]]
    assert m, f"triton_pin: no config of {t.base_fn.__name__} matches {w}"
    return m[0]


class _PinnedCache(dict):
    """Autotuner.cache stand-in: recorded keys -> recorded pick, every other key -> default (never benchmarks)."""
    def __init__(self, items, default):
        super().__init__(items); self.default = default

    def __contains__(self, k):
        return True

    def __missing__(self, k):
        return self.default


def apply(want, all_tuners=None):
    """Apply a pin dict to every matching autotuner (also used by kda_real_accuracy to switch pick sets)."""
    done = {}
    for name, t in (all_tuners or list(tuners())):
        if name not in want:
            continue
        w = want[name]
        if "keys" in w:
            items = {tuple(k): _match(t, c) for k, c in w["keys"]}
            t.configs = list(t.configs)   # keep all configs (keys pick among them)
            t.cache = _PinnedCache(items, _match(t, w["default"]))
        else:
            t.configs = [_match(t, w)]; t.cache = {}
        done[name] = w
    missing = set(want) - set(done)
    assert not missing, f"triton_pin: kernels not found: {missing}"
    return done


def install(path=None):
    path = path or os.environ.get("QWEN38_TRITON_PIN")
    if not path:
        return {}
    PINNED.update(apply(json.load(open(path))))
    print(f" -- triton_pin: pinned {sorted(PINNED)} from {path}", flush=True)
    return PINNED
