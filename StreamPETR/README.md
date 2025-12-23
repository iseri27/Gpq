We provide an example for StreamPETR.

# 1. Install StreamPETR

Follow [exiawsh/StreamPETR](https://github.com/exiawsh/StreamPETR) to install StreamPETR, prepare datasets and download checkpoints.

# 2. Clone Repo

```bash
git clone https://github.com/iseri27/Gpq
```

# 3. Replace Files

```bash
cd Gpq
mkdir -p $StreamPETR/projects/{configs/Gpq,mmdet3d_plugin/core/hook}

cp StreamPETR/projects/configs/Gpq/streampetr_r50_flash_900_300q_6+18e.py $StreamPETR/projects/configs/Gpq/
cp StreamPETR/projects/mmdet3d_plugin/core/hook/query_drop_hook.py $StreamPETR/projects/mmdet3d_plugin/core/hook
cp StreamPETR/projects/mmdet3d_plugin/models/dense_heads/streampetr_gpq_head.py $StreamPETR/projects/mmdet3d_plugin/models/dense_heads
cp SteramPETR/projects/mmdet3d_plugin/models/detectors/gpq_petr3d.py $StreamPETR/projects/mmdet3d_plugin/models/detectors
```

Note: `$StreamPETR` is the place where you install the StreamPETR project.

# 4. Add `import`

Add following import to `$StreamPETR/projects/mmdet3d_plugin/__init__.py`:

```python
from .core.hook.query_drop_hook import QueryDropHook
```

Add following import to `projects/mmdet3d_plugin/models/dense_heads/__init__.py`:

```python
from .streampetr_gpq_head import StreamPETRGpqHead
```

Add following import to `projects/mmdet3d_plugin/models/detectors/__init__.py`:

```python
from .gpq_petr3d import GpqPetr3D
```

# 5. Pruning

```bash
PYTHONPATH=$StreamPETR \
    python -m torch.distributed.launch \
        --nproc_per_node $YOUR_GPUS \
        --master_port $YOUR_PORT \
        tools/train.py \
        projects/configs/Gpq/streampetr_r50_flash_900_300q_6+18e.py \
        --launcher pytorch
```