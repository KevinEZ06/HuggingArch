#!/bin/bash
# tests/backend/inference/ 分片并行跑法。
#
# 这个套件在本机单进程跑不完：两次实测分别卡在 79% / 85%，被 90 分钟 timeout
# 杀掉，始终拿不到总结行。原因不是卡死而是真的慢，且慢的集中在后半段。
#
# 正确解法是分片并行，不是放宽 timeout。按文件轮转分片（`NR % N`）而不是按
# 目录切，因为慢用例在字母序上是聚集的——按前后切会让一片扛下绝大部分时间。
#
#   bash tests/run_inference_shards.sh            # 4 片，默认
#   SHARDS=8 bash tests/run_inference_shards.sh   # 8 片
#   SHARDS=4 PYTEST_ARGS=-rs bash tests/...       # 附加 pytest 参数
#
# 实测（4 片，2026-11-19）：
#   shard1  281 passed, 3 skipped   46:58
#   shard2  340 passed               30:35
#   shard3  393 passed               19:57
#   shard4  167 passed               14:56
#   合计 1181 passed, 3 skipped, 0 failed —— 墙钟 47 分钟
# `-e` is deliberately NOT set: a shard that fails must not stop the script
# before the others are waited on and summarised — reporting which shards
# failed is this script's whole job, and the failure is raised at the end.
# `pipefail` is, so a `cmd | tail` cannot report the exit of `tail`; that is
# exactly how a `kernel_bench run` exit of 3 once read as 0 here.
set -uo pipefail

SHARDS="${SHARDS:-4}"
OUT="${OUT:-/tmp/inference_shards}"
PYTEST_ARGS="${PYTEST_ARGS:-}"
PY="${PY:-.venv/bin/python}"
SUITE="${SUITE:-tests/backend/inference}"

cd "$(dirname "$0")/.." || exit 1
[ -x "$PY" ] || { echo "REFUSED: 找不到解释器 $PY"; exit 1; }
mkdir -p "$OUT"

files=$(ls "$SUITE"/test_*.py 2>/dev/null)
[ -n "$files" ] || { echo "REFUSED: $SUITE 下没有 test_*.py"; exit 1; }
total=$(echo "$files" | wc -l | tr -d ' ')
echo "套件 $SUITE: $total 个文件 -> $SHARDS 片，日志在 $OUT/"

pids=""
for i in $(seq 1 "$SHARDS"); do
  # NR%SHARDS 轮转：相邻文件进不同片，慢用例被摊开
  shard=$(echo "$files" | awk -v n="$SHARDS" -v i="$i" 'NR % n == i % n' | tr '\n' ' ')
  [ -n "$shard" ] || continue
  # shellcheck disable=SC2086
  "$PY" -m pytest -q -p no:cacheprovider $PYTEST_ARGS $shard \
    > "$OUT/shard$i.log" 2>&1 &
  pids="$pids $!"
  echo "  shard$i 已启动 (pid $!, $(echo $shard | wc -w | tr -d ' ') 个文件)"
done

echo
echo "等待全部完成..."
fail=0
for p in $pids; do wait "$p" || fail=1; done

echo
echo "===== 汇总 ====="
tp=0; ts=0; tf=0
for i in $(seq 1 "$SHARDS"); do
  [ -f "$OUT/shard$i.log" ] || continue
  line=$(grep -E "[0-9]+ (passed|failed|error)" "$OUT/shard$i.log" | tail -1) || line=""
  if [ -z "$line" ]; then
    # No summary line: killed by a timeout, or still running. Not zero
    # tests — unknown, which must not read as a clean shard.
    printf "  shard%-3s （无总结行，看 %s）\n" "$i" "$OUT/shard$i.log"
    fail=1
    continue
  fi
  printf "  shard%-3s %s\n" "$i" "$line"
  # pytest writes "281 passed, 3 skipped, 119 deselected in 2818s", so the
  # counts are comma-separated fields: split on the comma and anchor each
  # pattern, or a greedy `.*` eats all but the last digit ("281" read as
  # "1"). `sed -n` prints nothing and exits 0 for an outcome that did not
  # occur, which `pipefail` requires — `grep -o` would exit 1 and abort.
  n_of() {
    printf '%s\n' "$line" | tr ',' '\n' \
      | sed -nE "s/^ *([0-9]+) $1.*/\1/p" | head -1
  }
  n_passed=$(n_of passed); n_skipped=$(n_of skipped); n_failed=$(n_of failed)
  tp=$((tp + ${n_passed:-0}))
  ts=$((ts + ${n_skipped:-0}))
  tf=$((tf + ${n_failed:-0}))
done
echo "  ----"
echo "  合计 ${tp} passed, ${ts} skipped, ${tf} failed"

if [ "$tf" -gt 0 ] || [ "$fail" -ne 0 ]; then
  echo
  echo "有分片未通过，失败详情："
  grep -h -A3 "short test summary" "$OUT"/shard*.log 2>/dev/null | head -40
  exit 1
fi
