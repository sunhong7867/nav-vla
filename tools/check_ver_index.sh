#!/usr/bin/env bash
# docs/ver/*.md, docs/ver2/*.md 가 각자의 README.md 목록에 참조되는지 검사.
# 요약 문구의 품질은 사람 몫이고, 여기서는 "빠진 파일"만 기계적으로 잡는다.
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
fail=0
for dir in "$root/docs/ver" "$root/docs/ver2"; do
  [ -f "$dir/README.md" ] || continue
  for f in "$dir"/*.md; do
    base="$(basename "$f")"
    [ "$base" = "README.md" ] && continue
    if ! grep -q "$base" "$dir/README.md"; then
      echo "MISSING in ${dir#$root/}/README.md: $base"
      fail=1
    fi
    # 재현 명령어 블록(코드펜스) 없는 문서는 경고만 (README 필수 항목, 비차단)
    grep -q '^```' "$f" || echo "WARN no code block: ${f#$root/docs/}"
  done
done
[ "$fail" -eq 0 ] && echo "OK: all ver docs are referenced in their README."
exit "$fail"
