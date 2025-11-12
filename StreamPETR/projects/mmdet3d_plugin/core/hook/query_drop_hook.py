# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class QueryDropHook(Hook):
    """Set runner's epoch information to the model."""

    def __init__(
        self,
        interval=50,
        print_log=True,
        query_target=None,
        propagated_target=None,
    ):
        self.interval = interval
        self.print_log = print_log
        self.query_target = query_target
        self.propagated_target = propagated_target

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.interval == 0:
            runner.model.module.prune()
            num_query = runner.model.module.pts_bbox_head.num_query

            if self.print_log:
                print(f"current num_query: {num_query}")

                if self.propagated_target is None and self.query_target == num_query:
                    self.print_log = False

            if hasattr(runner.model.module.pts_bbox_head, "num_propagated"):
                num_propagated = runner.model.module.pts_bbox_head.num_propagated
                if self.print_log:
                    print(f"current num_propagated: {num_propagated}")

                    if (
                        self.propagated_target == num_propagated
                        and self.query_target == num_query
                    ):
                        self.print_log = False
