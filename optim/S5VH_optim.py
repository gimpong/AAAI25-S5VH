import torch.optim
from .basic import ScheduledOptim


class S5VH_opt_schedule(ScheduledOptim):
    def _schedule_step(self):
        self._schedule.step()
        if self.lr() < self.cfg.min_lr:
            self.set_lr(self.cfg.min_lr)