#!/bin/bash
echo "=== OS Info ==="
cat /etc/os-release 2>/dev/null | head -5
echo ""
echo "=== Package managers ==="
which apt-get 2>/dev/null && echo "apt-get found" || echo "apt-get NOT found"
which yum 2>/dev/null && echo "yum found" || echo "yum NOT found"
which dnf 2>/dev/null && echo "dnf found" || echo "dnf NOT found"
which conda 2>/dev/null && echo "conda found" || echo "conda NOT found"
which g++ 2>/dev/null && echo "g++ found" || echo "g++ NOT found"
which gcc 2>/dev/null && echo "gcc found" || echo "gcc NOT found"
which cc 2>/dev/null && echo "cc found" || echo "cc NOT found"
echo ""
echo "=== Try installing g++ via conda ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body
conda install -y gxx_linux-64 2>&1 | tail -10
echo ""
echo "=== Check g++ again ==="
which g++ 2>/dev/null && echo "g++ found" || echo "g++ NOT found"
which x86_64-conda-linux-gnu-g++ 2>/dev/null && echo "conda g++ found" || echo "conda g++ NOT found"
