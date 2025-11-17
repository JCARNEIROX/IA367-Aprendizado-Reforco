"""Gymnasium environment that exposes the GridFlex power-flow simulator.

The environment follows the standard Gymnasium interface so that any RL
algorithm (e.g. Stable-Baselines3, RLlib) can interact with the power-flow
framework step-by-step. Each ``step`` advances the OpenDSS simulation to the
next timestamp, applies the action as a BESS power setpoint, and returns an
observation built from demand history, BESS states of charge, and smoothing
indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import opendssdirect as dss
import pandas as pd
from gymnasium import spaces

from .bess import Bess, construct_bess, smoothness_index
from .generator import construct_generators
from .get_general_informations import get_informations
from .load import construct_lights, construct_loads
from .powerflow import power_flow_bess
from .read_spreadsheet import read_file_xlsx
from stable_baselines3.common.callbacks import BaseCallback
import os



@dataclass
class RewardBreakdown:
    """Stores the intermediate terms used to compute the reward."""

    delta_sigma: float = 0.0
    delta_norm: float = 0.0
    soc_penalty: float = 0.0

    @property
    def as_dict(self) -> Dict[str, float]:
        return {
            "delta_sigma": self.delta_sigma,
            "delta_norm": self.delta_norm,
            "soc_penalty": self.soc_penalty,
        }


class GridFlexEnv(gym.Env[np.ndarray, np.ndarray]):

    """Gymnasium environment that wraps the GridFlex framework.

    Parameters
    ----------
    config:
        Same dictionary used by ``modules.run``. Must include at least
        ``name_spreadsheet``, ``name_dss``, ``bess_bus``, ``past_values`` and
        ``seq_len``.
    reward_weights:
        Optional dictionary with the coefficients for ``sigma``, ``norm``,
        ``power`` (action magnitude) and ``soc`` (state-of-charge violation).

    Notes
    -----
    * The action space has one dimension per BESS unit and represents the
      requested active power in kW (positive = discharging).
    * The observation contains:
        ``[demand_history, bess_soc..., sigma, norm]``.
    * Call :meth:`episode_results` once an episode ends to convert the collected
      logs into Pandas DataFrames that mirror ``modules.run`` outputs.

    Example
    -------
    >>> env = GridFlexEnv(config)
    >>> obs, info = env.reset()
    >>> action = env.action_space.sample()
    >>> obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: Dict,
        reward_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.config = config.copy()
        self._validate_config()

        self.project_root = Path(__file__).resolve().parents[1]
        self.path_xlsx = self.project_root / "data" / "spreadsheets"
        self.path_dss = self.project_root / "data" / "dss_files"

        static_data = read_file_xlsx(str(self.path_xlsx / self.config["name_spreadsheet"]))
        self._general_df = static_data["General"]
        self._batteries_df = static_data["BESS"]
        self._generators_df = static_data["Generators"]
        self._loads_df = static_data["Loads"]
        self._lights_df = static_data["Public_Ilumination"]

        self.general_info = get_informations(self._general_df)
        self.file_dss = str(self.path_dss / self.config["name_dss"])

        # --- NOVO: permite sobrescrever start/end pelo config ---
        base_start = self.general_info.start_date
        base_end = self.general_info.end_date

        start_override = self.config.get("start_date", None)
        end_override = self.config.get("end_date", None)

        if start_override is not None:
            base_start = pd.to_datetime(start_override)

        if end_override is not None:
            base_end = pd.to_datetime(end_override)

        self.time_range = pd.date_range(
            base_start,
            base_end,
            freq=f"{self.general_info.timestep}T",
        ).to_list()
        # --------------------------------------------------------

        if len(self.time_range) == 0:
            raise ValueError("Time range is empty. Please check the spreadsheet dates.")

        self.interval_minutes = int(self.general_info.timestep)
        self.dt_hours = self.interval_minutes / 60.0

        self.history_len = max(1, int(self.config.get("past_values", 1)))
        self.seq_len = max(1, int(self.config.get("seq_len", self.history_len)))
        self.warmup_steps = max(self.history_len, self.seq_len)
        if self.warmup_steps >= len(self.time_range):
            raise ValueError("Warm-up steps exceed available timesteps in the dataset.")

        self.num_bess = len(self._batteries_df)
        if self.num_bess == 0:
            raise ValueError("At least one BESS entry is required in the spreadsheet.")

        self._pmax_array = self._batteries_df["Pmax"].to_numpy(dtype=float)
        self._soc_min_frac = (
            self._batteries_df["SOC_min(%)"].to_numpy(dtype=float) / 100.0
        )
        self._soc_max_frac = (
            self._batteries_df["SOC_max(%)"].to_numpy(dtype=float) / 100.0
        )

        obs_dim = self.history_len + self.num_bess + 2  # demand history + SoCs + sigma/norm
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-self._pmax_array.astype(np.float32),
            high=self._pmax_array.astype(np.float32),
            dtype=np.float32,
        )

        default_weights = {"delta_sigma": 1.0, "delta_norm": 1e-3, "soc": 10.0}
        if reward_weights is not None:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights

        self.bess_list: List[Bess] = []
        self.generators_list = []
        self.loads_list = []
        self.lights_list = []

        self.current_idx = 0
        self.demand_history: List[Sequence[float]] = []
        self.load_history: List[Sequence[float]] = []
        self.gen_history: List[Sequence[float]] = []
        self.loss_history: List[Sequence[float]] = []
        self.bus_power_history: List[Sequence[float]] = []
        self.bus_voltage_history: List[Sequence[float]] = []
        self.branch_history: List[Sequence[float]] = []
        self.bess_power_history: List[Sequence[float]] = []
        self.reward_trace: List[RewardBreakdown] = []
        self.action_trace: List[np.ndarray] = []
        self.sigma_history: List[float] = []
        self.norm_history: List[float] = []
        self.timestep_history: List[pd.Timestamp] = []

        self.latest_sigma = 0.0
        self.latest_norm = 0.0
        self.prev_sigma = 0.0
        self.prev_norm = 0.0

         # Snapshots do último episódio finalizado
        self.last_episode_indices: Optional[pd.DataFrame] = None
        self.last_episode_results: Optional[Dict[str, pd.DataFrame]] = None
        self.last_episode_rewards: Optional[List[RewardBreakdown]] = None

    # --------------------------------------------------------------------- #
    # Gym API                                                               #
    # --------------------------------------------------------------------- #

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        del options  # unused hook from Gymnasium

        self._reset_simulation_objects()
        self._reset_logs()
        self.current_idx = 0
        self.reward_trace.clear()
        self.action_trace.clear()
        self.latest_sigma = 0.0
        self.latest_norm = 0.0
        self.prev_sigma = 0.0
        self.prev_norm = 0.0

        self._prime_history()
        self.prev_sigma = self.latest_sigma
        self.prev_norm = self.latest_norm
        observation = self._build_observation()
        info = {"timestep": self.time_range[self.current_idx - 1]}
        return observation, info

    def step(self, action: np.ndarray):
        if self.current_idx >= len(self.time_range):
            raise RuntimeError("Episode already finished. Call reset() to start again.")

        action_vector = self._prepare_action(action)
        applied_action = self._apply_action_vector(action_vector)
        timestep = self.time_range[self.current_idx]
        (
            load,
            generation,
            bess,
            demand,
            losses,
            bus_power,
            bus_voltage,
            branch_df,
        ) = power_flow_bess(
            timestep,
            self.file_dss,
            self.bess_list,
            self.generators_list,
            self.loads_list,
            self.lights_list,
            dss,
        )

        self._log_step_results(
            load,
            generation,
            bess,
            demand,
            losses,
            bus_power,
            bus_voltage,
            branch_df,
        )
        self.current_idx += 1
        observation = self._build_observation()

        reward, breakdown = self._compute_reward(applied_action)
        self.reward_trace.append(breakdown)
        self.action_trace.append(applied_action)

        terminated = self.current_idx >= len(self.time_range)
        truncated = False

        if terminated:
            # Guarda cópias para uso fora (callback, pós-treino, etc.)
            self.last_episode_indices = self.indices_results()
            self.last_episode_results = self.episode_results()
            self.last_episode_rewards = list(self.reward_trace)

        info = {
            "timestep": timestep,
            "reward_terms": breakdown.as_dict,
            "sigma": self.latest_sigma,
            "norm": self.latest_norm,
            "soc": self._soc_array().copy(),
        }
        return observation, reward, terminated, truncated, info

    # --------------------------------------------------------------------- #
    # Public helpers                                                        #
    # --------------------------------------------------------------------- #

    def episode_results(self) -> Dict[str, pd.DataFrame]:
        """Return Pandas DataFrames with the same schema used in ``run``."""

        columns_bus = ["Timestep", "Bus", "P(kW)", "Q(kvar)"]
        columns_power = ["Timestep", "P(kW)", "Q(kvar)"]
        columns_branch = [
            "Timestep",
            "Branch",
            "Current(A)",
            "P(kW)",
            "Q(kvar)",
            "Losses(kW)",
        ]
        columns_bus_voltage = ["Timestep", "Bus", "Voltage (p.u.)"]
        columns_bess = ["Timestep", "Bess_Id", "P(kW)", "Q(kVar)", "E(kWh)", "SOC"]

        return {
            "bus_power": pd.DataFrame(self.bus_power_history, columns=columns_bus),
            "load": pd.DataFrame(self.load_history, columns=columns_power),
            "generation": pd.DataFrame(self.gen_history, columns=columns_power),
            "demand": pd.DataFrame(self.demand_history, columns=columns_power),
            "losses": pd.DataFrame(self.loss_history, columns=columns_power),
            "branch": pd.DataFrame(self.branch_history, columns=columns_branch),
            "bus_voltage": pd.DataFrame(self.bus_voltage_history, columns=columns_bus_voltage),
            "bess": pd.DataFrame(self.bess_power_history, columns=columns_bess),
        }

    def render(self):
        if not self.demand_history:
            print("Environment not stepped yet.")
            return
        timestep = self.time_range[self.current_idx - 1]
        demand_kw = self.demand_history[-1][1]
        soc = ", ".join(f"{val:.3f}" for val in self._soc_array())
        Pbess = self.bess_power_history[-1][2]
        # action = self.action_trace[-1][0]
        print(
            f"[{timestep}] Demand={demand_kw:.2f} kW "
            # f"Action {action:.2f} PBESS={Pbess:.2f} kW "
            f"PBESS={Pbess:.2f} kW "
            f"Sigma={self.latest_sigma:.4f} Norm={self.latest_norm:.4f} "
        )

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _validate_config(self) -> None:
        required = ["name_spreadsheet", "name_dss", "bess_bus"]
        missing = [field for field in required if field not in self.config]
        if missing:
            raise ValueError(f"Missing required config keys: {missing}")

    def _reset_simulation_objects(self) -> None:
        self.bess_list = construct_bess(self._batteries_df.copy())
        for bess in self.bess_list:
            bess.update_bus(self.config["bess_bus"])
        self.generators_list = construct_generators(self._generators_df.copy())
        self.loads_list = construct_loads(self._loads_df.copy())
        self.lights_list = construct_lights(self._lights_df.copy())

    def _reset_logs(self) -> None:
        self.load_history.clear()
        self.gen_history.clear()
        self.demand_history.clear()
        self.loss_history.clear()
        self.bus_power_history.clear()
        self.bus_voltage_history.clear()
        self.branch_history.clear()
        self.bess_power_history.clear()
        # limpa histórico de índices
        self.sigma_history.clear()
        self.norm_history.clear()
        self.timestep_history.clear()

    def _prime_history(self) -> None:
        warmup_actions = np.zeros(self.num_bess, dtype=np.float32)
        while len(self.demand_history) < self.warmup_steps:
            self._run_warmup_step(warmup_actions)

    def _run_warmup_step(self, action_vector: np.ndarray) -> None:
        timestep = self.time_range[self.current_idx]
        self._apply_action_vector(action_vector)
        (
            load,
            generation,
            bess,
            demand,
            losses,
            bus_power,
            bus_voltage,
            branch_df,
        ) = power_flow_bess(
            timestep,
            self.file_dss,
            self.bess_list,
            self.generators_list,
            self.loads_list,
            self.lights_list,
            dss,
        )
        self._log_step_results(
            load,
            generation,
            bess,
            demand,
            losses,
            bus_power,
            bus_voltage,
            branch_df,
        )
        self.current_idx += 1

    def _prepare_action(self, action: np.ndarray) -> np.ndarray:
        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_vector.size == 1 and self.num_bess == 1:
            return np.clip(action_vector, self.action_space.low, self.action_space.high)
        if action_vector.size != self.num_bess:
            raise ValueError(
                f"Action has shape {action_vector.shape}, "
                f"but {self.num_bess} BESS units are configured.",
            )
        return np.clip(action_vector, self.action_space.low, self.action_space.high)

    def _apply_action_vector(self, action_vector: np.ndarray) -> np.ndarray:
        applied = np.zeros_like(action_vector, dtype=np.float32)
        for idx, (bess, target_power) in enumerate(zip(self.bess_list, action_vector)):
            applied[idx] = self._apply_single_bess_action(bess, float(target_power))
        return applied

    def _apply_single_bess_action(self, bess: Bess, power_kw: float) -> float:
        """
        Aplica a ação em uma única bateria usando a lógica legada do framework.

        Parameters
        ----------
        bess : Bess
            Objeto da bateria.
        power_kw : float
            Potência pedida pelo agente (kW). Positiva = descarregar, negativa = carregar.

        Returns
        -------
        float
            Potência realmente aplicada (kW) na bateria.
        """

        # Mapeamento direto com o seu código
        PBessSeg = float(power_kw)        # potência "seguinte" pedida (kW)
        timestep = self.interval_minutes  # em minutos (mesma ideia do seu código)

        Soc = bess.SOC
        Et = bess.Et

        # Proteção básica (opcional, já há clipping em _prepare_action)
        PBessSeg = np.clip(PBessSeg, -bess.Pmax, bess.Pmax)

        if PBessSeg > 0:  # Bateria vai descarregar
            state = "DISCHARGING"

            # Limita pela potência máxima
            if PBessSeg > bess.Pmax:
                Pseg = bess.Pmax
            else:
                Pseg = PBessSeg

            # Energia no instante seguinte (descarga perde eficiência)
            Bess_E_seg = Et - Pseg * (timestep / 60.0) * (1.0 / bess.Efficiency)

            if Bess_E_seg < bess.Emin:
                # Ajusta potência para não passar abaixo de Emin
                Pseg = (Et - bess.Emin) / (timestep / 60.0)
                Bess_E_seg = bess.Emin

            # SOC seguinte
            Soc_next = Soc - (Pseg * (timestep / 60.0) * (1.0 / bess.Efficiency) / bess.Cmax)

            applied_kw = Pseg  # descarregando → potência aplicada é positiva

        else:  # Bateria vai carregar
            state = "CHARGING"

            # Na sua lógica, Pseg é a magnitude positiva da potência de carga
            if PBessSeg < -bess.Pmax:
                Pseg = bess.Pmax
            else:
                Pseg = -PBessSeg  # PBessSeg é negativo, então -PBessSeg > 0

            # Energia no instante seguinte (carga ganha energia com eficiência)
            Bess_E_seg = Et + Pseg * (timestep / 60.0) * bess.Efficiency

            if Bess_E_seg > bess.Emax:
                # Ajusta potência para não passar acima de Emax
                Pseg = (bess.Emax - Et) / (timestep / 60.0)
                Bess_E_seg = bess.Emax

            # SOC seguinte
            Soc_next = Soc + (Pseg * (timestep / 60.0) * bess.Efficiency / bess.Cmax)

            applied_kw = -Pseg  # carregando → potência aplicada é negativa

        # --------- Atualiza o objeto Bess (mantendo a interface do env) --------- #
        bess.Et = float(np.clip(Bess_E_seg, bess.Emin, bess.Emax))
        bess.SOC = float(np.clip(Soc_next, 0.0, 1.0))
        bess.Pt = float(applied_kw)
        bess.state = state if abs(applied_kw) > 0 else "IDLING"

        return float(applied_kw)

    def _log_step_results(
        self,
        load,
        generation,
        bess,
        demand,
        losses,
        bus_power,
        bus_voltage,
        branch_df,
    ) -> None:
        self.load_history.append(load)
        self.gen_history.append(generation)
        self.demand_history.append(demand)
        self.loss_history.append(losses)
        self.bus_power_history.extend(bus_power)
        self.bus_voltage_history.extend(bus_voltage)
        self.branch_history.extend(branch_df)
        self.bess_power_history.extend(bess)

        if len(self.demand_history) > 2:
            sigma, norm = smoothness_index([row[1] for row in self.demand_history])
            self.latest_sigma = float(sigma)
            self.latest_norm = float(norm)
        else:
            self.latest_sigma = 0.0
            self.latest_norm = 0.0

        current_ts = self.time_range[self.current_idx]

        self.timestep_history.append(current_ts)
        self.sigma_history.append(self.latest_sigma)
        self.norm_history.append(self.latest_norm)
        

    def _soc_array(self) -> np.ndarray:
        return np.array([bess.SOC for bess in self.bess_list], dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        demand_series = [row[1] for row in self.demand_history[-self.history_len :]]
        if len(demand_series) < self.history_len:
            pad_value = demand_series[0] if demand_series else 0.0
            demand_series = [pad_value] * (self.history_len - len(demand_series)) + demand_series
        obs = np.concatenate(
            [
                np.array(demand_series, dtype=np.float32),
                self._soc_array(),
                np.array([self.latest_sigma, self.latest_norm], dtype=np.float32),
            ]
        )
        return obs

    def _compute_reward(self, applied_action: np.ndarray) -> Tuple[float, RewardBreakdown]:
        delta_sigma = float(self.latest_sigma - self.prev_sigma)
        delta_norm = float(self.latest_norm - self.prev_norm)

        soc = self._soc_array()
        soc_violation = (
            np.maximum(0.0, self._soc_min_frac - soc)
            + np.maximum(0.0, soc - self._soc_max_frac)
        )
        soc_term = float(np.sum(soc_violation))

        breakdown = RewardBreakdown(
            delta_sigma=delta_sigma,
            delta_norm=delta_norm,
            soc_penalty=soc_term,
        )

        reward = -(
            self.reward_weights["delta_sigma"] * delta_sigma
            + self.reward_weights["delta_norm"] * delta_norm
            + self.reward_weights["soc"] * soc_term
        )
        self.prev_sigma = self.latest_sigma
        self.prev_norm = self.latest_norm
        return reward, breakdown

    def indices_results(self) -> pd.DataFrame:
        """Retorna um DataFrame com o histórico de sigma/norm ao longo do episódio."""
        return pd.DataFrame(
            {
                "Timestep": self.timestep_history,
                "Sigma": self.sigma_history,
                "Norm": self.norm_history,
            }
        )
    
## Callbacks
class RenderEveryNSteps(BaseCallback):
    def __init__(self, env, n: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.env_ref = env
        self.n = n

    def _on_step(self) -> bool:
        if self.n > 0 and self.num_timesteps % self.n == 0:
            self.env_ref.render()  # chama o render do seu GridFlexEnv
        return True

# Imprime Log e Salva Resultados a cada episódio
class TrainLoggingCallback(BaseCallback):
    """
    Callback que:
      - chama env.render() a cada `render_every_n` steps de treino;
      - ao final de cada episódio, salva:
          * índices (sigma/norm) via env.indices_results()
          * resultados de potência/carga via env.episode_results()
          * termos de recompensa via env.reward_trace
    """

    def __init__(self, env, save_dir: str, render_every_n: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.env_ref = env                # <- aqui guardamos o env passado
        self.save_dir = save_dir
        self.render_every_n = render_every_n
        self.episode_idx = 0

        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 1) Render periódico
        if self.render_every_n > 0 and self.num_timesteps % self.render_every_n == 0:
            self.env_ref.render()

        # 2) Salvamento no fim de episódio
        dones = self.locals.get("dones")
        if dones is not None and dones[0]:
            # Usa os snapshots preenchidos em step() quando terminated=True
            if (
                self.env_ref.last_episode_results is None
                or self.env_ref.last_episode_indices is None
                or self.env_ref.last_episode_rewards is None
            ):
                if self.verbose > 0:
                    print("[TrainLoggingCallback] Aviso: last_episode_* está None.")
                return True

            results = self.env_ref.last_episode_results
            indices_df = self.env_ref.last_episode_indices
            rewards_df = pd.DataFrame(
                [r.as_dict for r in self.env_ref.last_episode_rewards]
            )


            # salva índices
            indices_df.to_csv(
                os.path.join(self.save_dir, "indices_train.csv"),
                index=False,
            )

            # salva demand e bess (pode adicionar outros se quiser)
            results["demand"].to_csv(
                os.path.join(self.save_dir, "demand_train.csv"),
                index=False,
            )
            results["bess"].to_csv(
                os.path.join(self.save_dir, "bess_train.csv"),
                index=False,
            )

            # salva termos de recompensa
            rewards_df.to_csv(
                os.path.join(self.save_dir, "rewards_train.csv"),
                index=False,
            )

            if self.verbose > 0:
                print(
                    f"[TrainLoggingCallback] Episódio {self.episode_idx} salvo em {self.save_dir}"
                )

            self.episode_idx += 1

        return True