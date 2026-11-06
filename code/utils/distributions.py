import torch
from typing import Optional, Tuple
from torch.distributions import Categorical, constraints, Distribution
from torch.distributions.utils import broadcast_all
import numpy as np
import math
from numbers import Number

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)
HUGE_NEG = -1e8 #torch.tensor(-1e8, dtype=self.logits.dtype, device=device)

class MaskableCategorical(Categorical):
    def __init__(self, logits, masks = None):
        """
        Eliminate ("mask out") chosen categorical outcomes by setting their probability to 0.

        :param masks: An optional boolean ndarray of compatible shape with the distribution.
        If True, the corresponding choice's logit value is preserved. If False, it is set
        to a large negative value, resulting in near 0 probability. If masks is None, any
        previously applied masking is removed, and the original logits are restored.
        """
        self._original_logits = logits
        if masks is not None:
                device = self._original_logits.device
                self.masks = torch.as_tensor(masks, dtype=torch.bool, device=device).reshape(self._original_logits.shape)
                logits = torch.where(self.masks, self._original_logits, HUGE_NEG)
                super().__init__(logits=logits)
        else:
                self.masks = None
                super().__init__(logits=self._original_logits)
        

    def entropy(self) -> torch.Tensor:
        device = self.logits.device
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, 0.0)
        return -p_log_p.sum(-1)
      

class MaskableCategoricalDistribution:
    """
    Categorical distribution for discrete actions. Supports invalid action masking.

    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.distribution: Optional[MaskableCategorical] = None
        self.action_dim = action_dim

    def proba_distribution(
        self, action_logits: torch.Tensor, masks: Optional[torch.Tensor] = None):
        # Restructure shape to align with logits
        self.distribution = MaskableCategorical(logits=action_logits, masks=masks)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.log_prob(actions.clamp(max = self.action_dim - 1))

    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.entropy()

    def sample(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        assert self.distribution is not None, "Must set distribution parameters"
        return torch.argmax(self.distribution.probs, dim=1)

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        return self.sample() if not deterministic else self.mode()

def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,) for (n_batch, n_actions) input, scalar for (n_batch,) input
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=-1)
    else:
        tensor = tensor.sum()
    return tensor

# Source: https://github.com/toshas/torch_truncnorm/blob/main/TruncatedNormal.py
# BSD 3-Clause License

# Copyright (c) 2020, Anton Obukhov
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        # if self.a.dtype != self.b.dtype:
        #     raise ValueError('Truncation bounds types are different')
        # if any((self.a >= self.b).view(-1,).tolist()):
        #     raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return CONST_INV_SQRT_2PI *  torch.exp(-0.5 * x.square())

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + torch.erf(x * CONST_INV_SQRT_2))

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        log_prob = -0.5 * value.square() - self._log_Z + CONST_LOG_INV_SQRT_2PI
        return log_prob

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale

class DiagTruncatedGaussianDistribution:
    """
    Gaussian distribution with diagonal covariance matrix, for continuous actions.

    :param action_dim:  Dimension of the action space.
    """

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.distribution = None

    def proba_distribution(
        self, mean_actions: torch.Tensor, std: torch.Tensor
    ):
        """
        Create the distribution given its parameters (mean, std)

        :param mean_actions:
        :param log_std:
        :return:
        """
        self.distribution = TruncatedNormal(mean_actions, std, -1, 1)
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Get the log probabilities of actions according to the distribution.
        Note that you must first call the ``proba_distribution()`` method.

        :param actions:
        :return:
        """
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> Optional[torch.Tensor]:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> torch.Tensor:
        # Reparametrization trick to pass gradients
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # Update the proba distribution
        self.proba_distribution(mean_actions, log_std)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, mean_actions: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the log probability of taking an action
        given the distribution parameters.

        :param mean_actions:
        :param log_std:
        :return:
        """
        actions = self.actions_from_params(mean_actions, log_std)
        log_prob = self.log_prob(actions)
        return actions, log_prob
    
    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        """
        Return actions according to the probability distribution.

        :param deterministic:
        :return:
        """
        if deterministic:
            return self.mode()
        return self.sample()
    

if __name__ == "__main__":
    # test the distribution
    distribution = MaskableCategoricalDistribution(5)
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [-1.0, 1.0, 1.0, 1.0, 1.0]])
    distribution.proba_distribution(logits)
    print(distribution.log_prob(torch.tensor([0, 1])))
    distribution.apply_masking([[True, False, False, False, False], [False, True, False, False, False]])
    print(distribution.log_prob(torch.tensor([0, 1])))
