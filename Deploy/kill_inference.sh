#!/bin/bash
# 杀掉所有相关进程
pkill -9 -f demo.py 2>/dev/null
pkill -9 -f run_inference 2>/dev/null
pkill -9 -f "iopath" 2>/dev/null
pkill -9 -f "model_final" 2>/dev/null
sleep 2
echo "=== 剩余 python 进程 ==="
ps aux | grep python | grep -v grep | head -5
echo
echo "=== 剩余下载进程 ==="
ps aux | grep -iE "wget|curl|download" | grep -v grep | head -5
echo "DONE"
