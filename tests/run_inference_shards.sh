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
set -u

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
  line=$(grep -E "[0-9]+ (passed|failed|error)" "$OUT/shard$i.log" | tail -1)
  printf "  shard%-3s %s\n" "$i" "${line:-（无总结行，看 $OUT/shard$i.log）}"
  tp=$((tp + $(echo "$line" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)))
  ts=$((ts + $(echo "$line" | grep -oE '[0-9]+ skipped' | grep -oE '[0-9]+' || echo 0)))
  tf=$((tf + $(echo "$line" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)))
done
echo "  ----"
echo "  合计 ${tp} passed, ${ts} skipped, ${tf} failed"

if [ "$tf" -gt 0 ] || [ "$fail" -ne 0 ]; then
  echo
  echo "有分片未通过，失败详情："
  grep -h -A3 "short test summary" "$OUT"/shard*.log 2>/dev/null | head -40
  exit 1
fi
