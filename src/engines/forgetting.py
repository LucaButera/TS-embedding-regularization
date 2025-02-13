from math import e

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Bernoulli, Distribution

from env import rng
from src.nn.layers.embedding import IFNodeEmbedding


class ForgettingScheduler:
    """Forgetting scheduler for the node embeddings.
    All units are in steps.

    Args:
        emb_module: The node embedding module.
        stop_after: The number of steps after which
        the forgetting probability will be 0.
        Counting starts from after the warmup period.
        peak_prob: The peak forgetting probability at step 0.
        dampen_for: The number of steps for which the forgetting probability
        will be dampened after a reset.
        dampen_after: The number of times an embedding can be forgotten
        before the forgetting probability will be dampened.
        dampening_mode: The mode of dampening, either "linear", "exponential" or "none".
        max_forget: The maximum number of nodes to forget at each step,
        warmup_for: The number of steps to warm up the forgetting probability.
        warmup_mode: The mode of warmup, either "linear", "exponential" or "none".
        expressed as int of float fraction.
        step: The current step.
    """

    dampening_modes = ("linear", "exponential", "none")
    warmup_modes = ("linear", "none")
    exp_dampen_factor = 0.01  # ~1500 steps to dampen to 0

    def __init__(
        self,
        emb_module: IFNodeEmbedding,
        stop_after: int,
        peak_prob: float = 0.02,
        dampen_for: int = 100,
        dampen_after: int | None = None,
        dampening_mode: str | None = "linear",
        max_forget: int | float = 1.0,
        warmup_for: int = 0,
        warmup_mode: str | None = "none",
        step: int = 0,
    ):
        super().__init__()
        assert (
            emb_module.forgetting_strategy is not None
        ), "Embedding module must have a forgetting strategy."
        assert stop_after >= 0, "stop_after must be greater than or equal to 0."
        assert 0 <= peak_prob <= 1, "peak_prob must be in [0, 1]."
        assert dampen_for >= 0, "dampen_for must be greater than or equal to 0."
        assert (
            dampen_after is None or dampen_after >= 0
        ), "dampen_after must be greater than or equal to 0 or None."
        assert (
            dampening_mode is None or dampening_mode in self.dampening_modes
        ), f"dampening_mode must be one of {self.dampening_modes} or None."
        assert step >= 0, "step must be greater than or equal to 0."
        assert (isinstance(max_forget, float) and 0 <= max_forget <= 1) or (
            isinstance(max_forget, int) and 0 <= max_forget <= emb_module.n_nodes
        ), "max_forget must be a float in [0, 1] or an int in [0, n_nodes]."
        assert warmup_for >= 0, "warmup_for must be greater than or equal to 0."
        assert (
            warmup_mode is None or warmup_mode in self.warmup_modes
        ), f"warmup_mode must be one of {self.warmup_modes} or None."
        self.warmup_for = warmup_for
        self.stop_after = stop_after
        self.peak_prob = peak_prob
        self.dampen_for = dampen_for
        self.dampen_after = dampen_after if dampen_after is not None else torch.inf
        self.dampening_mode = dampening_mode
        self.warmup_mode = warmup_mode
        self.max_forget = (
            max_forget
            if isinstance(max_forget, int)
            else int(max_forget * emb_module.n_nodes)
        )
        self.unforgotten_for = torch.zeros(emb_module.n_nodes)
        self.times_forgotten = torch.zeros(emb_module.n_nodes)
        self.emb_module = emb_module
        self._current_step = step

    @property
    def current_decayed_p(self) -> float:
        if self.dampening_mode == "linear":
            decayed_p = (
                (self.stop_after - self._current_step + self.warmup_for)
                / self.stop_after
                * self.peak_prob
            )
            decayed_p = np.clip(decayed_p, 0, 1)
        elif self.dampening_mode == "exponential":
            decayed_p = self.peak_prob / (
                1
                + np.exp(
                    self.exp_dampen_factor
                    * (self._current_step - self.stop_after - self.warmup_for)
                )
            )
        else:
            decayed_p = (
                self.peak_prob
                if self._current_step <= self.stop_after + self.warmup_for
                else 0
            )
        return decayed_p

    @property
    def current_warmup_p(self) -> float:
        if self.warmup_mode == "linear":
            warmup_p = self.peak_prob * (self._current_step / self.warmup_for)
        else:
            warmup_p = 0
        return warmup_p

    @property
    def is_warmup(self) -> bool:
        return self._current_step < self.warmup_for

    @property
    def is_active(self) -> bool:
        return self._current_step <= self.stop_after + self.warmup_for

    @property
    def current_base_p(self) -> float:
        if self.is_warmup:
            base_p = self.current_warmup_p
        else:
            base_p = self.current_decayed_p
        return base_p

    @property
    def current_forget_dampening(self) -> float:
        if self.dampen_for > 0:
            return 1 - torch.exp(-self.unforgotten_for * e / self.dampen_for)
        else:
            return 1

    @property
    def current_overforget_dampening(self) -> float:
        return 1 / (1 + torch.exp(self.times_forgotten - self.dampen_after))

    @property
    def forgetting_distr(self) -> Distribution:
        return Bernoulli(
            self.current_base_p
            * self.current_forget_dampening
            * self.current_overforget_dampening
        )

    def step(
        self, step: int = None, return_mask: bool = False
    ) -> int | tuple[int, Tensor]:
        if step is not None:
            self._current_step = step
        to_forget = self.forgetting_distr.sample().to(torch.bool)
        n_forget = to_forget.sum().item()
        if n_forget > self.max_forget:
            forget_idx = to_forget.nonzero(as_tuple=True)[0]
            forget_idx = rng.choice(forget_idx, self.max_forget, replace=False)
            to_forget[forget_idx] = False
            n_forget = self.max_forget
        self.emb_module.forget(mask=to_forget.to(self.emb_module.emb.device))
        self.unforgotten_for += 1
        self.unforgotten_for[to_forget] = 0
        self.times_forgotten[to_forget] += 1
        self._current_step += 1
        if return_mask:
            return n_forget, to_forget
        else:
            return n_forget


class PeriodicForgettingScheduler(ForgettingScheduler):
    """Periodic forgetting scheduler for the node embeddings.
    Args:
        period: The period of forgetting in steps.
    """

    dampening_modes = ("none",)
    warmup_modes = ("none",)

    def __init__(
        self,
        emb_module: IFNodeEmbedding,
        stop_after: int,
        max_forget: int | float = 1.0,
        warmup_for: int = 0,
        step: int = 0,
        period: int = 100,
    ):
        assert period > 0, "period must be greater than 0."
        super().__init__(
            emb_module=emb_module,
            stop_after=stop_after,
            peak_prob=1.0,
            dampen_for=0,
            dampen_after=None,
            dampening_mode=None,
            max_forget=max_forget,
            warmup_for=warmup_for,
            warmup_mode=None,
            step=step,
        )
        self.period = period

    @property
    def current_decayed_p(self) -> float:
        if self.is_active and (self._current_step - self.warmup_for) % self.period == 0:
            return self.peak_prob
        else:
            return 0.0


schedulers = {
    "default": ForgettingScheduler,
    "periodic": PeriodicForgettingScheduler,
}
