# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class QueryDropHook(Hook):
    """Set runner's epoch information to the model."""

    def __init__(self, interval=50, query_target=None):
        self.interval = interval
        self.print_log = True
        self.query_target = query_target

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.interval == 0:
            runner.model.module.prune()
            num_query = runner.model.module.pts_bbox_head.num_query

            if self.print_log:
                print(f"current num_query: {num_query}")

                if num_query == self.query_target:
                    self.print_log = False
