"""G3 单步切分：把 4531 ms 的轨迹按采样算子切成单步，逐类统计设备时间。

边界取 `ArgMaxV2`（11 次，与 GreaterEqual/GatherV2/ReduceMean 同频——
采样链路每步一次）。段的身份由 attention 的 `Input Shapes` 判定：
prefill 的 query 是 1024 个 token，decode 的是 B=16 的单 token。

口径：一步之内所有下发的 Task Duration 之和，按类归并。这是设备上真正
执行的时间，与 wall 之间的差是主机气泡，单列不摊。
"""
import csv, glob, collections, json

F = glob.glob("/prof/*rank0*/PROF_*/mindstudio_profiler_output/op_summary_*.csv")[0]
rows = [r for r in csv.DictReader(open(F)) if r.get("Task Start Time(us)")]
for r in rows:
    r["_t"] = float(r["Task Start Time(us)"])
    r["_d"] = float(r["Task Duration(us)"] or 0)
rows.sort(key=lambda r: r["_t"])

bounds = [r["_t"] for r in rows if r["OP Type"] == "ArgMaxV2"]
print(f"边界算子 ArgMaxV2 {len(bounds)} 次 -> {len(bounds)} 段")

#: 昇腾算子名 -> HuggingArch 的类。从实测算子名读，不按常见命名猜:
#: 第一版按猜的名字归类时 other 占 84%,因为真实名字是
#: SparseFlashAttention / ChunkKdaFwd / QuantBatchMatmulV3 这些。
def cls(name):
    if "AicpuKernel" in name or name.startswith(
            ("hcom_", "allreduce", "alltoall", "allgather", "reducescatter",
             "broadcast", "send", "recv")):
        return "collective"
    if name in ("HcPre", "HcPost"):
        return "hyper_connection"
    if name in ("MatMulV3", "MatMulV2", "QuantBatchMatmulV3", "BatchMatMulV2"):
        return "gemm"
    if name.startswith("GroupedMatmul"):
        return "moe_grouped_matmul"
    if name in ("DynamicQuant", "AscendQuantV2", "AscendDequant"):
        return "dynamic_quant"
    if name in ("SparseFlashAttention", "ChunkKdaFwd", "PoolKeyIndexer",
                "FusedInferAttentionScore", "CausalConv1d",
                "IncreFlashAttention", "PromptFlashAttention"):
        return "attention"
    if name in ("RmsNorm", "AddRmsNorm", "Swiglu", "SwiGlu", "RotaryMul",
                "AddRmsNormQuant", "Add", "Mul"):
        return "norm_act"
    return "other"

segs = []
for i, start in enumerate(bounds):
    end = bounds[i + 1] if i + 1 < len(bounds) else rows[-1]["_t"] + 1
    seg = [r for r in rows if start <= r["_t"] < end]
    if not seg:
        continue
    # 这一段跑的是什么:看 attention 的输入形状
    shapes = [r.get("Input Shapes", "") for r in seg
              if cls(r["OP Type"]) == "attention"]
    kind = "?"
    for s in shapes:
        if "1024" in s:
            kind = "prefill S=1024"
            break
        if s:
            kind = "decode B=16"
    by = collections.defaultdict(float)
    for r in seg:
        by[cls(r["OP Type"])] += r["_d"]
    segs.append({"i": i, "kind": kind, "n": len(seg),
                 "span_ms": (max(r["_t"] + r["_d"] for r in seg) - start) / 1000,
                 "dev_ms": sum(by.values()) / 1000,
                 "by": {k: v / 1000 for k, v in by.items()}})

print(f"\n{'段':>3s} {'身份':16s} {'下发':>6s} {'设备 ms':>9s} {'墙上 ms':>9s}")
for s in segs:
    print(f"{s['i']:>3d} {s['kind']:16s} {s['n']:>6d} {s['dev_ms']:>9.2f} {s['span_ms']:>9.2f}")

CLASSES = ["collective", "attention", "moe_grouped_matmul", "gemm",
           "hyper_connection", "norm_act", "dynamic_quant", "other"]

def table(name, chosen):
    if not chosen:
        print(f"\n{name}: 无段"); return
    tot = sum(s["dev_ms"] for s in chosen) / len(chosen)
    print(f"\n=== {name}（{len(chosen)} 段均值，设备 {tot:.2f} ms）===")
    print(f"{'类':22s} {'ms':>9s} {'占比':>7s}")
    agg = collections.defaultdict(float)
    for s in chosen:
        for k, v in s["by"].items():
            agg[k] += v / len(chosen)
    for k in CLASSES:
        if agg.get(k):
            print(f"  {k:20s} {agg[k]:9.2f} {agg[k]/tot*100:6.1f}%")
    return {"mean_dev_ms": tot, "by": dict(agg), "segments": len(chosen)}

pre = [s for s in segs if s["kind"] == "prefill S=1024"]
dec = [s for s in segs if s["kind"] == "decode B=16"]
out = {"prefill": table("单步 Prefill (B=1, S=1024)", pre),
       "decode": table("单步 Decode (B=16, KV≈1024)", dec),
       "segments": segs}
# decode 步间一致性:切对了的证据
if len(dec) > 1:
    xs = [s["dev_ms"] for s in dec]
    print(f"\ndecode 步间: {min(xs):.2f}–{max(xs):.2f} ms，"
          f"极差 {(max(xs)-min(xs))/max(xs)*100:.1f}%")
open("/out/g3_steps.json", "w").write(json.dumps(out, indent=1))
print("\n写出 /out/g3_steps.json")
