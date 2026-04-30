#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate sam3d_body
python -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); [print("  GPU %d: %s, %.1fGB" % (i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_mem/1024**3)) for i in range(torch.cuda.device_count())]'
