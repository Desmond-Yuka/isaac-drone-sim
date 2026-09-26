#!/usr/bin/env bash
# 从 Isaac Sim 6.1 云端资产库下载无人机资产，保留原目录结构（文件间的相对引用不会断）
# 用法: bash download_assets.sh [资产目录前缀 ...]
set -u

BUCKET=https://omniverse-content-production.s3-us-west-2.amazonaws.com
ROOT=Assets/Isaac/6.1/Isaac
DEST="$(dirname "$(readlink -f "$0")")/assets"

if [ $# -eq 0 ]; then
  set -- Robots/NTNU/ARL-Robot-1/ Robots/Bitcraze/
fi

fail=0
for prefix in "$@"; do
  keys=$(curl -s "$BUCKET/?list-type=2&prefix=$ROOT/$prefix" | grep -oE "<Key>[^<]*</Key>" | sed -E 's#</?Key>##g' | grep -v '/\.thumbs/')
  for key in $keys; do
    rel=${key#"$ROOT/"}
    echo "下载 $rel"
    if ! curl -sf --create-dirs -o "$DEST/$rel" "$BUCKET/$key"; then
      echo "  -> 失败"
      fail=$((fail + 1))
    fi
  done
done

echo "=== 资产下载完成，失败 $fail 个 ==="
find "$DEST" -type f -printf '%10s  %P\n'
