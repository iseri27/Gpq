We provide an example for PETR.

# 1. Install PETR

Follow [megvii-research/PETR](https://github.com/megvii-research/PETR) to install PETR, prepare datasets and download checkpoints.

# 2. Clone Repo

```bash
git clone https://github.com/iseri27/Gpq
```

# 3. Replace Files

```bash
cd Gpq
mkdir -p $PETR/projects/{configs/Gpq,mmdet3d_plugin/core/hook}

cp PETR/projects/configs/Gpq/petr_r50_1408x512_900_300q_6+18e.py $PETR/projects/configs/Gpq/
cp PETR/projects/mmdet3d_plugin/core/hook/query_drop_hook.py $PETR/projects/mmdet3d_plugin/core/hook
cp PETR/projects/mmdet3d_plugin/models/dense_heads/petr_gpq_head.py $PETR/projects/mmdet3d_plugin/models/dense_heads
```

Note: `$PETR` is the place where you install the PETR project.

# 4. Add `import`

Add following import to `$PETR/projects/mmdet3d_plugin/__init__.py`:

```python
from .core.hook.query_drop_hook import QueryDropHook
```

Add following import to `projects/mmdet3d_plugin/models/dense_heads/__init__.py`:

```python
from .petr_gpq_head import PETRGpqHead
```

# 5. Pruning

```bash
PYTHONPATH=$PETR \
    python -m torch.distributed.launch \
        --nproc_per_node $YOUR_GPUS \
        --master_port $YOUR_PORT \
        tools/train.py \
        projects/configs/Gpq/petr_r50_1408x512_900_300q_6+18e.py \
        --launcher pytorch
```