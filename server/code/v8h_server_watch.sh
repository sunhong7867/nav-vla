#!/bin/bash
# 서버 상주: v8h 학습 감시 + 중간 체크포인트 실시간 프루닝(최신 2개 유지).
# 종료 시 060000만 남기고 ~/sunhong/nav-vla/logs/v8h_watch.status 에 결과 기록.
RUN=navvla_smolvla_v8h
D=$HOME/sunhong/nav-vla/runs/$RUN/checkpoints
LOG=$HOME/sunhong/nav-vla/logs/$RUN.log
ST=$HOME/sunhong/nav-vla/logs/v8h_watch.status
echo "RUNNING $(date +%F_%T)" > "$ST"
for i in $(seq 1 130); do
  if [ -d "$D" ]; then
    ls -d "$D"/0* 2>/dev/null | head -n -2 | grep -v 060000 | xargs -r rm -rf
  fi
  if [ -f "$D/060000/pretrained_model/model.safetensors" ]; then
    for c in "$D"/0*; do [ "$(basename "$c")" = 060000 ] || rm -rf "$c"; done
    echo "DONE $(date +%F_%T) $(du -sh $HOME/sunhong/nav-vla/runs/$RUN | cut -f1)" > "$ST"
    exit 0
  fi
  if grep -qiE 'Traceback|CUDA out of memory' "$LOG" 2>/dev/null; then
    echo "ERROR $(date +%F_%T)" > "$ST"; tail -5 "$LOG" >> "$ST"; exit 7
  fi
  sleep 300
done
echo "TIMEOUT $(date +%F_%T)" > "$ST"; exit 8
