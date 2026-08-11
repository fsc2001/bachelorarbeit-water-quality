import csv
import json
import os
import sys
import random

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    sync_envs_normalization,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "NeuralSurrogateKalmanChlorineEstimation"

MODEL_DIR = ROOT / "models" / "multiseed"
RESULT_DIR = ROOT / "results"

PLACEMENT_PATH = (
    RESULT_DIR
    / "ppo_sensor_placements.json"
)

with PLACEMENT_PATH.open(
    "r",
    encoding="utf-8",
) as file:
    PPO_SENSOR_PLACEMENTS = json.load(file)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO))

from Env.network_config import NETWORKS
from Env.chlorine_env import (
    ChlorineControlEnv,
    create_network_scenario,
)


NETWORK_NAME = "cydbp"
NETWORK = NETWORKS[NETWORK_NAME]

# ============================================================
# Experiment configuration
# ============================================================

RUN_NAME = "final_seeded_ppo_20000"
TRAINING_SEEDS = [0, 1, 2, 3, 4]


VARIANTS = {
    "ekf_random": {
        "use_estimated_state": True,
        "placement": "random",
    },
    "ekf_centrality": {
        "use_estimated_state": True,
        "placement": "centrality",
    },
    "oracle": {
        "use_estimated_state": False,
        "placement": None,
    },
}


TOTAL_TIMESTEPS = 5000
EVALUATION_STEPS = 500

VALIDATION_SEED = 100
VALIDATION_FREQ = 1000
VALIDATION_STEPS = 200

EVALUATION_SEED = 200

LOWER_CHLORINE_BOUND = 0.3
UPPER_CHLORINE_BOUND = 2.0

CHLORINE_PENALTY_WEIGHT = 0.01

SKIP_EXISTING_MODELS = True


# ============================================================
# Environment creation
# ============================================================

def get_sensor_indices(
    placement_name: Optional[str],
):
    if placement_name is None:
        return None, None

    placement = (
        PPO_SENSOR_PLACEMENTS[
            NETWORK_NAME
        ][placement_name]
    )

    node_indices = placement[
        "node_indices"
    ]

    link_indices = placement[
        "link_indices"
    ]

    return node_indices, link_indices
def set_environment_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
def make_env(
    use_estimated_state: bool,
    seed: int,
    placement_name: Optional[str],
) -> Callable[[], Monitor]:

    def _init() -> Monitor:
        set_environment_seed(seed)
        os.chdir(REPO)

        (
            node_indices,
            link_indices,
        ) = get_sensor_indices(
            placement_name
        )

        if node_indices is None:
            n_sensors = 0
        else:
            n_sensors = len(
                node_indices
            )

        scenario_config, _ = create_network_scenario(
            network_config=NETWORK,
            n_sensors=n_sensors,
            uncertainty_seed=seed,
        )

        env = ChlorineControlEnv(
            scenario_config=scenario_config,
            network_config=NETWORK,
            n_sensors=n_sensors,
            use_estimated_state=(
                use_estimated_state
            ),
            chlorine_penalty_weight=(
                CHLORINE_PENALTY_WEIGHT
            ),
            sensor_node_indices=(
                node_indices
            ),
            sensor_link_indices=(
                link_indices
            ),
        )

        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return Monitor(env)

    return _init

def create_training_env(
    use_estimated_state: bool,
    seed: int,
    placement_name: Optional[str],
) -> VecNormalize:

    base_env = DummyVecEnv(
        [
            make_env(
                use_estimated_state=use_estimated_state,
                seed=seed,
                placement_name=placement_name,
            )
        ]
    )

    return VecNormalize(
        base_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )


def create_validation_env(
    use_estimated_state: bool,
    seed: int,
    placement_name: Optional[str],
) -> VecNormalize:

    base_env = DummyVecEnv(
        [
            make_env(
                use_estimated_state=use_estimated_state,
                seed=seed,
                placement_name=placement_name,
            )
        ]
    )

    env = VecNormalize(
        base_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99,
    )

    env.training = False
    env.norm_reward = False

    return env


# ============================================================
# Paths
# ============================================================

def get_last_model_paths(
    variant_name: str,
    training_seed: int,
) -> Tuple[Path, Path]:

    model_path = (
        MODEL_DIR
        / (
            f"{NETWORK_NAME}_ppo_{variant_name}"
            f"_seed{training_seed}_{RUN_NAME}"
        )
    )

    vecnormalize_path = (
        MODEL_DIR
        / (
            f"{NETWORK_NAME}_ppo_{variant_name}"
            f"_seed{training_seed}_{RUN_NAME}"
            "_vecnormalize.pkl"
        )
    )

    return model_path, vecnormalize_path


def get_best_model_paths(
    variant_name: str,
    training_seed: int,
) -> Tuple[Path, Path, Path]:

    save_dir = (
        MODEL_DIR
        / "best_models"
        / (
            f"{NETWORK_NAME}_ppo_{variant_name}"
            f"_seed{training_seed}_{RUN_NAME}"
        )
    )

    model_path = save_dir / "best_model"
    vecnormalize_path = save_dir / "best_vecnormalize.pkl"

    return save_dir, model_path, vecnormalize_path


# ============================================================
# Fixed-step validation callback
# ============================================================

class FixedStepValidationCallback(BaseCallback):
    """
    Evaluates the current policy after a fixed number of
    training steps and saves the best checkpoint.

    Lower validation scores are better.
    """

    def __init__(
        self,
        validation_env: VecNormalize,
        validation_seed: int,
        eval_freq: int,
        eval_steps: int,
        save_dir: Path,
        chlorine_penalty_weight: float,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)

        self.validation_env = validation_env
        self.validation_seed = validation_seed
        self.eval_freq = eval_freq
        self.eval_steps = eval_steps
        self.save_dir = save_dir

        self.chlorine_penalty_weight = (
            chlorine_penalty_weight
        )

        self.best_score = np.inf
        self.validation_number = 0

        self.log_path = (
            self.save_dir
            / "validation_log.csv"
        )

    def _init_callback(self) -> None:
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.log_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "validation_number",
                    "training_timesteps",
                    "score",
                    "mean_reward",
                    "mean_action",
                    "cumulative_action",
                    "outside_range_fraction",
                    "lower_violation_fraction",
                    "upper_violation_fraction",
                    "mean_violation_per_node_value",
                    "cumulative_total_violation",
                    "minimum_node_chlorine",
                    "maximum_node_chlorine",
                    "is_new_best",
                ]
            )

    def _run_validation(self) -> Dict[str, float]:
        # Validierung verwendet dieselben
        # Normalisierungsstatistiken wie das Training.
        sync_envs_normalization(
            self.training_env,
            self.validation_env,
        )

        self.validation_env.training = False
        self.validation_env.norm_reward = False

        self.validation_env.seed(
            self.validation_seed
        )

        obs = self.validation_env.reset()

        rewards: List[float] = []
        actions: List[float] = []

        total_node_values = 0

        lower_violation_count = 0
        upper_violation_count = 0

        cumulative_lower_violation = 0.0
        cumulative_upper_violation = 0.0

        minimum_node_chlorine = np.inf
        maximum_node_chlorine = -np.inf

        for _ in range(self.eval_steps):
            action, _ = self.model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, done, infos = (
                self.validation_env.step(action)
            )

            info = infos[0]

            reward_value = float(
                np.asarray(reward).reshape(-1)[0]
            )

            action_value = float(
                info.get(
                    "physical_chlorine_action",
                    np.asarray(action).reshape(-1)[0],
                )
            )

            if "scada_data" not in info:
                raise RuntimeError(
                    "Validation info does not contain "
                    "'scada_data'."
                )

            node_quality = np.asarray(
                info[
                    "scada_data"
                ].get_data_nodes_quality(),
                dtype=np.float64,
            ).reshape(-1)

            if not np.all(np.isfinite(node_quality)):
                raise RuntimeError(
                    "Validation produced non-finite "
                    "chlorine concentrations."
                )

            lower_mask = (
                node_quality
                < LOWER_CHLORINE_BOUND
            )

            upper_mask = (
                node_quality
                > UPPER_CHLORINE_BOUND
            )

            lower_violation = np.maximum(
                LOWER_CHLORINE_BOUND
                - node_quality,
                0.0,
            )

            upper_violation = np.maximum(
                node_quality
                - UPPER_CHLORINE_BOUND,
                0.0,
            )

            rewards.append(reward_value)
            actions.append(action_value)

            total_node_values += node_quality.size

            lower_violation_count += int(
                np.sum(lower_mask)
            )

            upper_violation_count += int(
                np.sum(upper_mask)
            )

            cumulative_lower_violation += float(
                np.sum(lower_violation)
            )

            cumulative_upper_violation += float(
                np.sum(upper_violation)
            )

            minimum_node_chlorine = min(
                minimum_node_chlorine,
                float(np.min(node_quality)),
            )

            maximum_node_chlorine = max(
                maximum_node_chlorine,
                float(np.max(node_quality)),
            )

            if bool(done[0]):
                self.validation_env.seed(
                    self.validation_seed
                )

                obs = self.validation_env.reset()

        cumulative_total_violation = (
            cumulative_lower_violation
            + cumulative_upper_violation
        )

        mean_action = float(
            np.mean(actions)
        )

        mean_violation_per_step = (
            cumulative_total_violation
            / self.eval_steps
        )

        score = (
            mean_violation_per_step
            + self.chlorine_penalty_weight
            * mean_action
        )

        return {
            "score": float(score),

            "mean_reward": float(
                np.mean(rewards)
            ),

            "mean_action": mean_action,

            "cumulative_action": float(
                np.sum(actions)
            ),

            "outside_range_fraction": (
                lower_violation_count
                + upper_violation_count
            ) / total_node_values,

            "lower_violation_fraction": (
                lower_violation_count
                / total_node_values
            ),

            "upper_violation_fraction": (
                upper_violation_count
                / total_node_values
            ),

            "mean_violation_per_node_value": (
                cumulative_total_violation
                / total_node_values
            ),

            "cumulative_total_violation": (
                cumulative_total_violation
            ),

            "minimum_node_chlorine": (
                minimum_node_chlorine
            ),

            "maximum_node_chlorine": (
                maximum_node_chlorine
            ),
        }

    def _save_best_model(
        self,
        metrics: Dict[str, float],
    ) -> None:

        model_path = (
            self.save_dir
            / "best_model"
        )

        normalization_path = (
            self.save_dir
            / "best_vecnormalize.pkl"
        )

        metrics_path = (
            self.save_dir
            / "best_validation_metrics.json"
        )

        self.model.save(
            str(model_path)
        )

        normalization_env = (
            self.model.get_vec_normalize_env()
        )

        if normalization_env is None:
            raise RuntimeError(
                "Training environment does not "
                "contain VecNormalize."
            )

        normalization_env.save(
            str(normalization_path)
        )

        output = {
            "training_timesteps": int(
                self.num_timesteps
            ),
            **metrics,
        }

        with metrics_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output,
                file,
                indent=2,
            )

    def _append_log(
        self,
        metrics: Dict[str, float],
        is_new_best: bool,
    ) -> None:

        with self.log_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    self.validation_number,
                    self.num_timesteps,
                    metrics["score"],
                    metrics["mean_reward"],
                    metrics["mean_action"],
                    metrics["cumulative_action"],
                    metrics[
                        "outside_range_fraction"
                    ],
                    metrics[
                        "lower_violation_fraction"
                    ],
                    metrics[
                        "upper_violation_fraction"
                    ],
                    metrics[
                        "mean_violation_per_node_value"
                    ],
                    metrics[
                        "cumulative_total_violation"
                    ],
                    metrics[
                        "minimum_node_chlorine"
                    ],
                    metrics[
                        "maximum_node_chlorine"
                    ],
                    is_new_best,
                ]
            )

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        self.validation_number += 1

        metrics = self._run_validation()

        is_new_best = (
            metrics["score"]
            < self.best_score
        )

        if is_new_best:
            self.best_score = metrics["score"]

            self._save_best_model(
                metrics
            )

        self._append_log(
            metrics=metrics,
            is_new_best=is_new_best,
        )

        self.logger.record(
            "validation/score",
            metrics["score"],
        )

        self.logger.record(
            "validation/outside_range_fraction",
            metrics["outside_range_fraction"],
        )

        self.logger.record(
            "validation/mean_action",
            metrics["mean_action"],
        )

        if self.verbose:
            status = (
                "NEW BEST"
                if is_new_best
                else "not improved"
            )

            print("\n" + "-" * 72)

            print(
                f"VALIDATION AT "
                f"{self.num_timesteps} STEPS"
            )

            print("-" * 72)

            print(
                f"Score: "
                f"{metrics['score']:.6f}"
            )

            print(
                "Outside range: "
                f"{metrics['outside_range_fraction']:.6f}"
            )

            print(
                "Mean violation per node value: "
                f"{metrics['mean_violation_per_node_value']:.6f}"
            )

            print(
                f"Mean action: "
                f"{metrics['mean_action']:.6f}"
            )

            print(
                f"Status: {status}"
            )

        return True


# ============================================================
# Training
# ============================================================

def train_variant(
    variant_name: str,
    use_estimated_state: bool,
    placement_name: Optional[str],
    training_seed: int,
) -> Tuple[Path, Path]:

    last_model_path, last_vecnormalize_path = (
        get_last_model_paths(
            variant_name=variant_name,
            training_seed=training_seed,
        )
    )

    (
        best_model_dir,
        best_model_path,
        best_vecnormalize_path,
    ) = get_best_model_paths(
        variant_name=variant_name,
        training_seed=training_seed,
    )

    last_model_zip_path = Path(
        str(last_model_path) + ".zip"
    )

    best_model_zip_path = Path(
        str(best_model_path) + ".zip"
    )

    all_files_exist = (
        last_model_zip_path.exists()
        and last_vecnormalize_path.exists()
        and best_model_zip_path.exists()
        and best_vecnormalize_path.exists()
    )

    if (
        SKIP_EXISTING_MODELS
        and all_files_exist
    ):
        print("\n" + "=" * 72)

        print(
            f"ÜBERSPRINGE TRAINING: "
            f"{variant_name}, Seed {training_seed}"
        )

        print("=" * 72)

        print(
            "Bestes Modell existiert bereits:",
            best_model_zip_path,
        )

        return (
            best_model_path,
            best_vecnormalize_path,
        )

    print("\n" + "=" * 72)

    print(
        f"TRAINING: {variant_name}, "
        f"Seed {training_seed}"
    )

    print("=" * 72)

    env = create_training_env(
        use_estimated_state=use_estimated_state,
        seed=training_seed,
        placement_name=placement_name,
    )

    validation_env = create_validation_env(
        use_estimated_state=use_estimated_state,
        seed=VALIDATION_SEED,
        placement_name=placement_name,
    )

    callback = FixedStepValidationCallback(
        validation_env=validation_env,
        validation_seed=VALIDATION_SEED,
        eval_freq=VALIDATION_FREQ,
        eval_steps=VALIDATION_STEPS,
        save_dir=best_model_dir,
        chlorine_penalty_weight=(
            CHLORINE_PENALTY_WEIGHT
        ),
        verbose=1,
    )

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=training_seed,
            device="cpu",
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            progress_bar=False,
        )

        # Zusätzlich zum besten Checkpoint wird das
        # letzte Modell gespeichert.
        model.save(
            str(last_model_path)
        )

        env.save(
            str(last_vecnormalize_path)
        )

        print("\nTraining abgeschlossen.")

        print(
            "Letztes Modell:",
            last_model_path,
        )

        print(
            "Bestes Modell:",
            best_model_path,
        )

    finally:
        env.close()
        validation_env.close()

    if not best_model_zip_path.exists():
        raise RuntimeError(
            "No best model was saved. "
            "Check VALIDATION_FREQ and TOTAL_TIMESTEPS."
        )

    if not best_vecnormalize_path.exists():
        raise RuntimeError(
            "No VecNormalize file was saved "
            "for the best model."
        )

    return (
        best_model_path,
        best_vecnormalize_path,
    )


# ============================================================
# Final test evaluation
# ============================================================

def evaluate_variant(
    variant_name: str,
    use_estimated_state: bool,
    placement_name: Optional[str],
    training_seed: int,
    model_path: Path,
    vecnormalize_path: Path,
) -> Dict[str, float]:

    set_environment_seed(
        EVALUATION_SEED
    )

    base_env = DummyVecEnv(
        [
            make_env(
                use_estimated_state=use_estimated_state,
                seed=EVALUATION_SEED,
                placement_name=placement_name,
            )
        ]
    )

    env = VecNormalize.load(
        str(vecnormalize_path),
        base_env,
    )

    env.training = False
    env.norm_reward = False

    env.seed(
        EVALUATION_SEED
    )

    model = PPO.load(
        str(model_path),
        env=env,
        device="cpu",
    )

    rewards: List[float] = []
    actions: List[float] = []

    total_node_values = 0

    lower_violation_count = 0
    upper_violation_count = 0

    total_lower_violation = 0.0
    total_upper_violation = 0.0

    steps_with_lower_violation = 0
    steps_with_upper_violation = 0
    steps_with_any_violation = 0

    minimum_node_chlorine = np.inf
    maximum_node_chlorine = -np.inf

    best_metrics_path = (
        model_path.parent
        / "best_validation_metrics.json"
    )

    selected_training_timesteps = -1

    if best_metrics_path.exists():
        with best_metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            best_metrics = json.load(file)

        selected_training_timesteps = int(
            best_metrics[
                "training_timesteps"
            ]
        )

    print("\n" + "=" * 72)

    print(
        f"TEST EVALUATION: {variant_name}, "
        f"Training-Seed {training_seed}"
    )

    print(
        "Selected checkpoint:",
        selected_training_timesteps,
    )

    print("=" * 72)

    try:
        set_environment_seed(
            EVALUATION_SEED
        )

        env.seed(
            EVALUATION_SEED
        )

        obs = env.reset()

        for step_idx in range(
            EVALUATION_STEPS
        ):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, done, infos = (
                env.step(action)
            )

            info = infos[0]

            reward_value = float(
                np.asarray(reward).reshape(-1)[0]
            )

            action_value = float(
                info.get(
                    "physical_chlorine_action",
                    np.asarray(action).reshape(-1)[0],
                )
            )

            if "scada_data" not in info:
                raise RuntimeError(
                    "Evaluation info does not contain "
                    "'scada_data'."
                )

            node_quality = np.asarray(
                info[
                    "scada_data"
                ].get_data_nodes_quality(),
                dtype=np.float64,
            ).reshape(-1)

            if not np.all(np.isfinite(node_quality)):
                raise RuntimeError(
                    "Non-finite chlorine values in "
                    f"evaluation step {step_idx + 1}."
                )

            lower_mask = (
                node_quality
                < LOWER_CHLORINE_BOUND
            )

            upper_mask = (
                node_quality
                > UPPER_CHLORINE_BOUND
            )

            lower_violation = np.maximum(
                LOWER_CHLORINE_BOUND
                - node_quality,
                0.0,
            )

            upper_violation = np.maximum(
                node_quality
                - UPPER_CHLORINE_BOUND,
                0.0,
            )

            rewards.append(reward_value)
            actions.append(action_value)

            total_node_values += node_quality.size

            lower_violation_count += int(
                np.sum(lower_mask)
            )

            upper_violation_count += int(
                np.sum(upper_mask)
            )

            total_lower_violation += float(
                np.sum(lower_violation)
            )

            total_upper_violation += float(
                np.sum(upper_violation)
            )

            has_lower_violation = bool(
                np.any(lower_mask)
            )

            has_upper_violation = bool(
                np.any(upper_mask)
            )

            if has_lower_violation:
                steps_with_lower_violation += 1

            if has_upper_violation:
                steps_with_upper_violation += 1

            if (
                has_lower_violation
                or has_upper_violation
            ):
                steps_with_any_violation += 1

            minimum_node_chlorine = min(
                minimum_node_chlorine,
                float(np.min(node_quality)),
            )

            maximum_node_chlorine = max(
                maximum_node_chlorine,
                float(np.max(node_quality)),
            )

            if step_idx < 5:
                print(
                    f"Step {step_idx + 1:03d}: "
                    f"action={action_value:.4f}, "
                    f"reward={reward_value:.4f}, "
                    f"Cl min={np.min(node_quality):.4f}, "
                    f"Cl max={np.max(node_quality):.4f}"
                )

            if bool(done[0]):
                env.seed(
                    EVALUATION_SEED
                )

                obs = env.reset()

        total_violation = (
            total_lower_violation
            + total_upper_violation
        )

        result = {
            "variant": variant_name,
            "training_seed": training_seed,
            "validation_seed": VALIDATION_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "selected_training_timesteps": (
                selected_training_timesteps
            ),
            "evaluation_steps": EVALUATION_STEPS,

            "mean_action": float(
                np.mean(actions)
            ),

            "std_action": float(
                np.std(actions)
            ),

            "min_action": float(
                np.min(actions)
            ),

            "max_action": float(
                np.max(actions)
            ),

            "cumulative_action": float(
                np.sum(actions)
            ),

            "mean_reward": float(
                np.mean(rewards)
            ),

            "cumulative_reward": float(
                np.sum(rewards)
            ),

            "lower_violation_fraction": (
                lower_violation_count
                / total_node_values
            ),

            "upper_violation_fraction": (
                upper_violation_count
                / total_node_values
            ),

            "outside_range_fraction": (
                lower_violation_count
                + upper_violation_count
            ) / total_node_values,

            "temporal_lower_fraction": (
                steps_with_lower_violation
                / EVALUATION_STEPS
            ),

            "temporal_upper_fraction": (
                steps_with_upper_violation
                / EVALUATION_STEPS
            ),

            "temporal_any_fraction": (
                steps_with_any_violation
                / EVALUATION_STEPS
            ),

            "cumulative_lower_violation": (
                total_lower_violation
            ),

            "cumulative_upper_violation": (
                total_upper_violation
            ),

            "cumulative_total_violation": (
                total_violation
            ),

            "mean_violation_per_node_value": (
                total_violation
                / total_node_values
            ),

            "minimum_node_chlorine": (
                minimum_node_chlorine
            ),

            "maximum_node_chlorine": (
                maximum_node_chlorine
            ),
        }

        print("\nTest result:")

        print(
            "Outside range:",
            result[
                "outside_range_fraction"
            ],
        )

        print(
            "Mean violation:",
            result[
                "mean_violation_per_node_value"
            ],
        )

        print(
            "Cumulative action:",
            result[
                "cumulative_action"
            ],
        )

        print(
            "Mean reward:",
            result["mean_reward"],
        )

        return result

    finally:
        env.close()


# ============================================================
# Result storage
# ============================================================

def save_raw_results(
    results: List[Dict[str, float]],
) -> Path:

    output_path = (
        RESULT_DIR / f"{NETWORK_NAME}_ppo_{RUN_NAME}_raw.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    return output_path


def calculate_std(
    values: np.ndarray,
) -> float:

    if values.size < 2:
        return float("nan")

    return float(
        np.std(values, ddof=1)
    )


def save_summary(
    results: List[Dict[str, float]],
) -> Path:

    output_path = (
        RESULT_DIR / f"{NETWORK_NAME}_ppo_{RUN_NAME}_summary.csv"
    )

    excluded_fields = {
        "variant",
        "training_seed",
        "validation_seed",
        "evaluation_seed",
    }

    numeric_metrics = [
        key
        for key, value
        in results[0].items()
        if (
            key not in excluded_fields
            and isinstance(
                value,
                (int, float, np.number),
            )
        )
    ]

    summary_rows = []

    variants = sorted(
        {
            row["variant"]
            for row in results
        }
    )

    for variant_name in variants:
        variant_results = [
            row
            for row in results
            if row["variant"] == variant_name
        ]

        summary_row = {
            "variant": variant_name,
            "n_training_seeds": len(
                variant_results
            ),
        }

        for metric in numeric_metrics:
            values = np.asarray(
                [
                    row[metric]
                    for row in variant_results
                ],
                dtype=np.float64,
            )

            summary_row[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            summary_row[
                f"{metric}_std"
            ] = calculate_std(values)

        summary_rows.append(
            summary_row
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    return output_path


def print_summary(
    results: List[Dict[str, float]],
) -> None:

    important_metrics = [
        "outside_range_fraction",
        "lower_violation_fraction",
        "upper_violation_fraction",
        "mean_violation_per_node_value",
        "cumulative_action",
        "mean_reward",
    ]

    variants = sorted(
        {
            row["variant"]
            for row in results
        }
    )

    print("\n" + "=" * 72)
    print("SUMMARY OVER TRAINING SEEDS")
    print("=" * 72)

    for variant_name in variants:
        variant_results = [
            row
            for row in results
            if row["variant"] == variant_name
        ]

        print(
            f"\n{variant_name.upper()}"
        )

        for metric in important_metrics:
            values = np.asarray(
                [
                    row[metric]
                    for row in variant_results
                ],
                dtype=np.float64,
            )

            mean = float(
                np.mean(values)
            )

            std = calculate_std(
                values
            )

            if np.isnan(std):
                std_text = "n/a"
            else:
                std_text = f"{std:.6f}"

            print(
                f"{metric}: "
                f"{mean:.6f} ± {std_text}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: List[
        Dict[str, float]
    ] = []

    for (
            variant_name,
            variant_config,
    ) in VARIANTS.items():

        use_estimated_state = (
            variant_config[
                "use_estimated_state"
            ]
        )

        placement_name = (
            variant_config[
                "placement"
            ]
        )

        for training_seed in TRAINING_SEEDS:
            (
                model_path,
                vecnormalize_path,
            ) = train_variant(
                variant_name=variant_name,
                use_estimated_state=(
                    use_estimated_state
                ),
                training_seed=training_seed,
                placement_name=placement_name,
            )

            result = evaluate_variant(
                variant_name=variant_name,
                use_estimated_state=use_estimated_state,
                placement_name=placement_name,
                training_seed=training_seed,
                model_path=model_path,
                vecnormalize_path=vecnormalize_path,
            )

            results.append(
                result
            )

            save_raw_results(
                results
            )

    raw_path = save_raw_results(
        results
    )

    summary_path = save_summary(
        results
    )

    print_summary(
        results
    )

    print("\nFiles saved:")
    print(raw_path)
    print(summary_path)
    import winsound

    winsound.MessageBeep()

if __name__ == "__main__":
    main()
