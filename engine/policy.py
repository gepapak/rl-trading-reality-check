from __future__ import annotations

import collections
from functools import partial
from typing import Any, Generator, NamedTuple, Optional, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn
from torch.distributions import Beta

from stable_baselines3.common.distributions import Distribution, sum_independent_dims
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo import PPO


class RuleBasedPolicy:
    """
    Minimal non-learning policy wrapper for deterministic controllers.
    """

    def __init__(self, agent_name: str, observation_space: Any, action_space: Any, env: Any):
        self.agent_name = agent_name
        self.mode = "RULE"
        self.observation_space = observation_space
        self.action_space = action_space
        self.env = env
        self.device = "cpu"
        self.callback = None
        self.num_timesteps = 0

    def _resolve_env(self) -> Any:
        env = self.env
        for attr in ("base_env", "env"):
            try:
                next_env = getattr(env, attr, None)
                if next_env is not None:
                    env = next_env
            except Exception:
                pass
        return env

    def predict(self, obs, deterministic: bool = True):
        del deterministic
        env = self._resolve_env()
        action = None
        if env is not None and hasattr(env, "get_rule_based_agent_action"):
            action = env.get_rule_based_agent_action(self.agent_name, obs)
        if action is None:
            shape = getattr(self.action_space, "shape", (1,))
            action = np.zeros(shape, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        return action, None


class BetaDistribution(Distribution):
    """
    Beta distribution on [-1, 1] obtained by affine-transforming a Beta(0, 1).
    """

    def __init__(self, action_dim: int, epsilon: float = 1e-6):
        super().__init__()
        self.action_dim = int(action_dim)
        self.epsilon = float(epsilon)
        self.alpha = None
        self.beta = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        return nn.Linear(latent_dim, 2 * self.action_dim)

    def proba_distribution(self, action_logits: th.Tensor) -> "BetaDistribution":
        alpha_raw, beta_raw = th.chunk(action_logits, 2, dim=1)
        alpha = th.nn.functional.softplus(alpha_raw) + self.epsilon
        beta = th.nn.functional.softplus(beta_raw) + self.epsilon
        self.alpha = alpha
        self.beta = beta
        self.distribution = Beta(alpha, beta)
        return self

    def proba_distribution_from_concentrations(
        self,
        alpha: th.Tensor,
        beta: th.Tensor,
    ) -> "BetaDistribution":
        """Construct the bounded distribution from already-positive parameters."""
        self.alpha = th.clamp(alpha, min=self.epsilon)
        self.beta = th.clamp(beta, min=self.epsilon)
        self.distribution = Beta(self.alpha, self.beta)
        return self

    def _to_unit_interval(self, actions: th.Tensor) -> th.Tensor:
        scaled = 0.5 * (actions + 1.0)
        return th.clamp(scaled, self.epsilon, 1.0 - self.epsilon)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        unit_actions = self._to_unit_interval(actions)
        log_prob = self.distribution.log_prob(unit_actions) - np.log(2.0)
        return sum_independent_dims(log_prob)

    def entropy(self) -> Optional[th.Tensor]:
        return sum_independent_dims(self.distribution.entropy() + np.log(2.0))

    def sample(self) -> th.Tensor:
        unit = self.distribution.rsample()
        return 2.0 * unit - 1.0

    def mode(self) -> th.Tensor:
        mean = self.distribution.mean
        return 2.0 * mean - 1.0

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class BetaActorCriticPolicy(ActorCriticPolicy):
    """
    SB3-compatible actor-critic policy that uses a Beta action distribution for
    bounded continuous actions instead of a Gaussian.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[nn.Module] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        beta_epsilon: float = 1e-6,
    ):
        self.beta_epsilon = float(beta_epsilon)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)  # type: ignore[arg-type, return-value]
        data.update(
            dict(
                beta_epsilon=self.beta_epsilon,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def _build(self, lr_schedule: Schedule) -> None:
        self._build_mlp_extractor()
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_dist = BetaDistribution(get_action_dim(self.action_space), epsilon=self.beta_epsilon)
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.get_distribution(observation).get_actions(deterministic=deterministic)


class CentralizedRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    central_observations: th.Tensor
    action_masks: th.Tensor


class CentralizedCriticRolloutBuffer(RolloutBuffer):
    """Rollout buffer with a side-channel for MAPPO centralized critic observations."""

    def __init__(self, *args, central_obs_dim: int = 0, **kwargs):
        self.central_obs_dim = int(max(1, central_obs_dim))
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        super().reset()
        self.central_observations = np.zeros(
            (self.buffer_size, self.n_envs, self.central_obs_dim),
            dtype=np.float32,
        )
        # Default to active so non-investor MAPPO policies preserve standard
        # PPO behavior. The meta-controller writes the investor decision mask.
        self.action_masks = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)

    def add_central_observation(self, pos: int, central_obs: Any) -> None:
        arr = np.asarray(central_obs, dtype=np.float32).reshape(-1)
        if arr.size != self.central_obs_dim:
            raise ValueError(
                f"central_obs_dim mismatch: expected {self.central_obs_dim}, got {arr.size}"
            )
        self.central_observations[int(pos), 0, :] = arr

    def add_action_mask(self, pos: int, action_mask: float) -> None:
        value = float(action_mask)
        if not np.isfinite(value):
            raise ValueError("action_mask must be finite")
        self.action_masks[int(pos), 0] = float(1.0 if value > 0.5 else 0.0)

    def get(self, batch_size: Optional[int] = None) -> Generator[CentralizedRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for tensor in [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "central_observations",
                "action_masks",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> CentralizedRolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.central_observations[batch_inds],
            self.action_masks[batch_inds].flatten(),
        )
        return CentralizedRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class _CentralizedCriticMixin:
    """Training-only centralized value path; actor/predict still consume local observations."""

    def _setup_centralized_critic(
        self,
        central_obs_dim: int,
        central_net_arch: Optional[list[int]],
        activation_fn: type[nn.Module],
    ) -> None:
        self.central_obs_dim = int(max(1, central_obs_dim))
        self.central_net_arch = list(central_net_arch or [256, 128, 64])
        layers: list[nn.Module] = []
        last_dim = self.central_obs_dim
        for width in self.central_net_arch:
            layers.append(nn.Linear(last_dim, int(width)))
            layers.append(activation_fn())
            last_dim = int(width)
        self.central_value_mlp = nn.Sequential(*layers)
        self.central_value_net = nn.Linear(last_dim, 1)
        self.uses_centralized_critic = True

    def predict_central_values(self, central_obs: th.Tensor) -> th.Tensor:
        if not isinstance(central_obs, th.Tensor):
            central_obs = th.as_tensor(central_obs, dtype=th.float32, device=self.device)
        central_obs = central_obs.to(self.device).float()
        if central_obs.dim() == 1:
            central_obs = central_obs.unsqueeze(0)
        return self.central_value_net(self.central_value_mlp(central_obs))

    def evaluate_actions_centralized(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        central_obs: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.predict_central_values(central_obs)
        entropy = distribution.entropy()
        return values, log_prob, entropy


class CentralizedCriticActorCriticPolicy(_CentralizedCriticMixin, ActorCriticPolicy):
    """ActorCriticPolicy with a MAPPO centralized critic and local actor input."""

    def __init__(
        self,
        *args,
        central_obs_dim: int = 0,
        central_net_arch: Optional[list[int]] = None,
        **kwargs,
    ):
        self._mappo_central_obs_dim_arg = int(max(1, central_obs_dim))
        self._mappo_central_net_arch_arg = list(central_net_arch or [256, 128, 64])
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self._setup_centralized_critic(
            self._mappo_central_obs_dim_arg,
            self._mappo_central_net_arch_arg,
            self.activation_fn,
        )
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "central_obs_dim": int(getattr(self, "central_obs_dim", self._mappo_central_obs_dim_arg)),
                "central_net_arch": list(getattr(self, "central_net_arch", self._mappo_central_net_arch_arg)),
            }
        )
        return data


class CentralizedCriticBetaActorCriticPolicy(_CentralizedCriticMixin, BetaActorCriticPolicy):
    """Beta actor policy with a MAPPO centralized critic and local actor input."""

    def __init__(
        self,
        *args,
        central_obs_dim: int = 0,
        central_net_arch: Optional[list[int]] = None,
        **kwargs,
    ):
        self._mappo_central_obs_dim_arg = int(max(1, central_obs_dim))
        self._mappo_central_net_arch_arg = list(central_net_arch or [256, 128, 64])
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        self._setup_centralized_critic(
            self._mappo_central_obs_dim_arg,
            self._mappo_central_net_arch_arg,
            self.activation_fn,
        )
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "central_obs_dim": int(getattr(self, "central_obs_dim", self._mappo_central_obs_dim_arg)),
                "central_net_arch": list(getattr(self, "central_net_arch", self._mappo_central_net_arch_arg)),
            }
        )
        return data


class CentralizedCriticForecastMirrorBetaActorCriticPolicy(
    CentralizedCriticBetaActorCriticPolicy
):
    """MAPPO actor mirrored toward a causal forecast expert in Beta space.

    The learned actor and the forecast expert are combined as an exact
    product of Beta densities.  A separate state-dependent trust network
    controls the forecast exponent.  PPO updates the base actor and the
    centralized critic; the trust network is detached from PPO and is updated
    only with strictly matured counterfactual utility gradients supplied by
    the environment.
    """

    def __init__(
        self,
        *args,
        forecast_mirror_prior_concentration: float = 8.0,
        forecast_mirror_trust_max: float = 1.0,
        forecast_mirror_trust_hidden_dim: int = 32,
        forecast_mirror_trust_initial: float = 0.10,
        forecast_mirror_trust_regularization: float = 0.01,
        forecast_mirror_trust_min_samples: int = 32,
        forecast_mirror_trust_batch_size: int = 256,
        forecast_mirror_trust_gradient_clip: float = 5.0,
        forecast_mirror_counterfactual_learning: bool = True,
        forecast_mirror_mask_base_actor_forecast: bool = False,
        **kwargs,
    ):
        self.forecast_mirror_prior_concentration = float(
            max(forecast_mirror_prior_concentration, 0.0)
        )
        self.forecast_mirror_trust_max = float(max(forecast_mirror_trust_max, 0.0))
        self.forecast_mirror_trust_hidden_dim = int(
            max(forecast_mirror_trust_hidden_dim, 1)
        )
        self.forecast_mirror_trust_initial = float(
            np.clip(forecast_mirror_trust_initial, 1e-4, 1.0 - 1e-4)
        )
        self.forecast_mirror_trust_regularization = float(
            max(forecast_mirror_trust_regularization, 0.0)
        )
        self.forecast_mirror_trust_min_samples = int(
            max(forecast_mirror_trust_min_samples, 1)
        )
        self.forecast_mirror_trust_batch_size = int(
            max(forecast_mirror_trust_batch_size, 1)
        )
        self.forecast_mirror_trust_gradient_clip = float(
            max(forecast_mirror_trust_gradient_clip, 1e-6)
        )
        self.forecast_mirror_counterfactual_learning = bool(
            forecast_mirror_counterfactual_learning
        )
        # Corrected-CFM (default off = legacy CFM behaviour).
        self.forecast_mirror_mask_base_actor_forecast = bool(
            forecast_mirror_mask_base_actor_forecast
        )
        self._mirror_trust_samples: collections.deque[tuple[np.ndarray, float]] = (
            collections.deque(maxlen=8192)
        )
        self._last_mirror_diagnostics: dict[str, th.Tensor] = {}
        self._last_mirror_train_diagnostics: dict[str, float] = {}
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        obs_dim = int(np.prod(self.observation_space.shape))
        hidden = int(self.forecast_mirror_trust_hidden_dim)
        self.forecast_mirror_trust_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            self.activation_fn(),
            nn.Linear(hidden, 1),
        )
        first_layer = self.forecast_mirror_trust_net[0]
        if isinstance(first_layer, nn.Linear):
            nn.init.orthogonal_(first_layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(first_layer.bias)
        final_layer = self.forecast_mirror_trust_net[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)
        initial_logit = float(
            np.log(self.forecast_mirror_trust_initial)
            - np.log1p(-self.forecast_mirror_trust_initial)
        )
        if isinstance(final_layer, nn.Linear):
            nn.init.constant_(final_layer.bias, initial_logit)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "forecast_mirror_prior_concentration": self.forecast_mirror_prior_concentration,
                "forecast_mirror_trust_max": self.forecast_mirror_trust_max,
                "forecast_mirror_trust_hidden_dim": self.forecast_mirror_trust_hidden_dim,
                "forecast_mirror_trust_initial": self.forecast_mirror_trust_initial,
                "forecast_mirror_trust_regularization": self.forecast_mirror_trust_regularization,
                "forecast_mirror_trust_min_samples": self.forecast_mirror_trust_min_samples,
                "forecast_mirror_trust_batch_size": self.forecast_mirror_trust_batch_size,
                "forecast_mirror_trust_gradient_clip": self.forecast_mirror_trust_gradient_clip,
                "forecast_mirror_counterfactual_learning": self.forecast_mirror_counterfactual_learning,
                "forecast_mirror_mask_base_actor_forecast": self.forecast_mirror_mask_base_actor_forecast,
            }
        )
        return data

    def _base_concentrations(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, _ = self.mlp_extractor(features)
        else:
            pi_features, _ = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
        action_logits = self.action_net(latent_pi)
        alpha_raw, beta_raw = th.chunk(action_logits, 2, dim=1)
        return (
            th.nn.functional.softplus(alpha_raw) + self.beta_epsilon,
            th.nn.functional.softplus(beta_raw) + self.beta_epsilon,
        )

    def _trust(self, obs: th.Tensor) -> th.Tensor:
        flat = obs.float().reshape(obs.shape[0], -1)
        return self.forecast_mirror_trust_max * th.sigmoid(
            self.forecast_mirror_trust_net(flat)
        )

    def _forecast_mirror_distribution(
        self,
        obs: th.Tensor,
        *,
        detach_trust: bool,
    ) -> BetaDistribution:
        flat = obs.float().reshape(obs.shape[0], -1)
        if flat.shape[1] < 22:
            raise RuntimeError(
                "Forecast-mirror MAPPO requires the 22D investor observation contract"
            )
        if self.forecast_mirror_mask_base_actor_forecast:
            # Dual-view: the base MAPPO actor sees market state plus execution
            # geometry (dims 17-19) but NOT forecast-derived dims (12-16, 20-21),
            # so the forecast enters only through the trust-gated mirror below.
            # The trust network still receives the full 22D observation.
            base_flat = flat.clone()
            base_flat[:, 12:17] = 0.0
            base_flat[:, 20:22] = 0.0
            base_alpha, base_beta = self._base_concentrations(base_flat.reshape(obs.shape))
        else:
            base_alpha, base_beta = self._base_concentrations(obs)
        prior_coordinate = th.clamp(flat[:, 20:21], -1.0, 1.0)
        evidence = th.clamp(flat[:, 21:22], 0.0, 1.0)
        prior_unit = 0.5 * (prior_coordinate + 1.0)
        concentration = self.forecast_mirror_prior_concentration
        prior_alpha = 1.0 + concentration * prior_unit
        prior_beta = 1.0 + concentration * (1.0 - prior_unit)
        trust = self._trust(obs)
        trust_for_distribution = trust.detach() if detach_trust else trust
        exponent = evidence * trust_for_distribution
        final_alpha = base_alpha + exponent * (prior_alpha - 1.0)
        final_beta = base_beta + exponent * (prior_beta - 1.0)
        self._last_mirror_diagnostics = {
            "base_alpha": base_alpha.detach(),
            "base_beta": base_beta.detach(),
            "prior_alpha": prior_alpha.detach(),
            "prior_beta": prior_beta.detach(),
            "final_alpha": final_alpha.detach(),
            "final_beta": final_beta.detach(),
            "prior_coordinate": prior_coordinate.detach(),
            "evidence": evidence.detach(),
            "trust": trust.detach(),
            "exponent": (evidence * trust).detach(),
        }
        return self.action_dist.proba_distribution_from_concentrations(
            final_alpha, final_beta
        )

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        if not isinstance(obs, th.Tensor):
            raise TypeError("Forecast-mirror policy requires a tensor Box observation")
        return self._forecast_mirror_distribution(obs, detach_trust=False)

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        distribution = self._forecast_mirror_distribution(obs, detach_trust=False)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.predict_values(obs)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self.get_distribution(observation).get_actions(deterministic=deterministic)

    def evaluate_actions(
        self, obs: PyTorchObs, actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        if not isinstance(obs, th.Tensor):
            raise TypeError("Forecast-mirror policy requires a tensor Box observation")
        distribution = self._forecast_mirror_distribution(obs, detach_trust=True)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()

    def evaluate_actions_centralized(
        self,
        obs: PyTorchObs,
        actions: th.Tensor,
        central_obs: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        if not isinstance(obs, th.Tensor):
            raise TypeError("Forecast-mirror policy requires a tensor Box observation")
        distribution = self._forecast_mirror_distribution(obs, detach_trust=True)
        return (
            self.predict_central_values(central_obs),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_mirror_diagnostics(self) -> dict[str, np.ndarray]:
        return {
            key: value.detach().cpu().numpy().copy()
            for key, value in self._last_mirror_diagnostics.items()
        }

    def add_mirror_trust_samples(self, samples: list[dict[str, Any]]) -> None:
        expected = int(np.prod(self.observation_space.shape))
        for sample in samples:
            obs = np.asarray(sample.get("observation", []), dtype=np.float32).reshape(-1)
            gradient = float(sample.get("trust_gradient", 0.0))
            if obs.size != expected or not np.isfinite(gradient):
                continue
            self._mirror_trust_samples.append((obs.copy(), gradient))

    def train_mirror_trust(self) -> dict[str, float]:
        count = len(self._mirror_trust_samples)
        if not self.forecast_mirror_counterfactual_learning:
            self._mirror_trust_samples.clear()
            diag = {"sample_count": float(count), "updated": 0.0, "disabled": 1.0}
            self._last_mirror_train_diagnostics = diag
            return diag
        if count < self.forecast_mirror_trust_min_samples:
            diag = {"sample_count": float(count), "updated": 0.0}
            self._last_mirror_train_diagnostics = diag
            return diag
        batch_count = min(count, self.forecast_mirror_trust_batch_size)
        samples = [self._mirror_trust_samples.popleft() for _ in range(batch_count)]
        obs_np = np.stack([item[0] for item in samples], axis=0)
        gradients_np = np.asarray([item[1] for item in samples], dtype=np.float32)
        scale = float(max(np.mean(np.abs(gradients_np)), 1e-8))
        normalized = np.clip(
            gradients_np / scale,
            -self.forecast_mirror_trust_gradient_clip,
            self.forecast_mirror_trust_gradient_clip,
        )
        obs = th.as_tensor(obs_np, dtype=th.float32, device=self.device)
        gradients = th.as_tensor(normalized, dtype=th.float32, device=self.device).reshape(-1, 1)
        trust = self._trust(obs)
        loss = -(gradients.detach() * trust).mean()
        loss = loss + self.forecast_mirror_trust_regularization * th.mean(trust.square())
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(
            self.forecast_mirror_trust_net.parameters(),
            self.forecast_mirror_trust_gradient_clip,
        )
        self.optimizer.step()
        diag = {
            "sample_count": float(batch_count),
            "updated": 1.0,
            "loss": float(loss.detach().cpu().item()),
            "trust_mean": float(trust.detach().mean().cpu().item()),
            "gradient_mean": float(np.mean(gradients_np)),
            "gradient_abs_mean": float(np.mean(np.abs(gradients_np))),
            "queue_remaining": float(len(self._mirror_trust_samples)),
        }
        self._last_mirror_train_diagnostics = diag
        return diag


class CentralizedCriticPPO(PPO):
    """PPO train loop using centralized critic batches when the policy provides them."""

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        actor_active_fractions = []
        continue_training = True
        last_loss = th.zeros((), device=self.device)

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                if hasattr(self.policy, "evaluate_actions_centralized") and hasattr(rollout_data, "central_observations"):
                    # mappo-2x2 task
                    values, log_prob, entropy = self.policy.evaluate_actions_centralized(
                        rollout_data.observations,
                        actions,
                        rollout_data.central_observations,
                    )
                else:
                    values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                action_mask = getattr(
                    rollout_data,
                    "action_masks",
                    th.ones_like(advantages),
                ).flatten() > 0.5
                actor_active_fractions.append(float(action_mask.float().mean().item()))
                active_count = int(action_mask.sum().item())
                if self.normalize_advantage and active_count > 1:
                    active_advantages = advantages[action_mask]
                    advantages = (advantages - active_advantages.mean()) / (
                        active_advantages.std() + 1e-8
                    )

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                if active_count > 0:
                    policy_loss = -th.min(policy_loss_1, policy_loss_2)[action_mask].mean()
                else:
                    policy_loss = 0.0 * log_prob.sum()
                pg_losses.append(policy_loss.item())
                clip_fraction = (
                    th.mean((th.abs(ratio[action_mask] - 1) > clip_range).float()).item()
                    if active_count > 0
                    else 0.0
                )
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if active_count > 0:
                    entropy_loss = (
                        -th.mean(-log_prob[action_mask])
                        if entropy is None
                        else -th.mean(entropy[action_mask])
                    )
                else:
                    entropy_loss = 0.0 * log_prob.sum()
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                last_loss = loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean(((th.exp(log_ratio) - 1) - log_ratio)[action_mask]).cpu().numpy()
                        if active_count > 0
                        else np.asarray(0.0, dtype=np.float32)
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses) if entropy_losses else 0.0)
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses) if pg_losses else 0.0)
        self.logger.record("train/value_loss", np.mean(value_losses) if value_losses else 0.0)
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs) if approx_kl_divs else 0.0)
        self.logger.record("train/clip_fraction", np.mean(clip_fractions) if clip_fractions else 0.0)
        self.logger.record(
            "train/actor_active_fraction",
            np.mean(actor_active_fractions) if actor_active_fractions else 1.0,
        )
        self.logger.record("train/loss", last_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if hasattr(self.policy, "train_mirror_trust"):
            mirror_diag = self.policy.train_mirror_trust()
            for key, value in mirror_diag.items():
                self.logger.record(f"train/forecast_mirror_{key}", value)


__all__ = [
    "RuleBasedPolicy",
    "BetaDistribution",
    "BetaActorCriticPolicy",
    "CentralizedCriticActorCriticPolicy",
    "CentralizedCriticBetaActorCriticPolicy",
    "CentralizedCriticForecastMirrorBetaActorCriticPolicy",
    "CentralizedCriticPPO",
    "CentralizedCriticRolloutBuffer",
]
