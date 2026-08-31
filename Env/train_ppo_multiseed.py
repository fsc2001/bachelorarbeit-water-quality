"""
This module trains and evaluates PPO controllers for different sensor configurations.
"""

import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    sync_envs_normalization,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_REPO = PROJECT_DIR / "NeuralSurrogateKalmanChlorineEstimation"
MODEL_DIR = PROJECT_DIR / "models" / "multiseed"
RESULTS_DIR = PROJECT_DIR / "results"

sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REFERENCE_REPO))

from Env.chlorine_env import ChlorineControlEnv, create_network_scenario
from Env.network_config import NETWORKS


NETWORK_NAME = "cydbp"
NETWORK = NETWORKS[NETWORK_NAME]

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

TOTAL_TIMESTEPS = 20000

VALIDATION_SEED = 100
VALIDATION_FREQ = 2000
VALIDATION_STEPS = 500

EVALUATION_SEED = 200
EVALUATION_STEPS = 1000

LOWER_CHLORINE_BOUND = 0.3
UPPER_CHLORINE_BOUND = 2.0
CHLORINE_PENALTY_WEIGHT = 0.01

SKIP_EXISTING_MODELS = True

SUMMARY_METRICS = [
    "outside_range_fraction",
    "lower_violation_fraction",
    "upper_violation_fraction",
    "mean_violation_per_node_value",
    "cumulative_action",
    "mean_reward",
]


def load_sensor_placements():
    placement_path = RESULTS_DIR / "ppo_sensor_placements.json"

    with placement_path.open("r", encoding="utf-8") as file:
        return json.load(file)


PPO_SENSOR_PLACEMENTS = load_sensor_placements()


def get_sensor_indices(placement_name):
    if placement_name is None:
        return None, None

    placement = PPO_SENSOR_PLACEMENTS[NETWORK_NAME][placement_name]

    return (
        placement["node_indices"],
        placement["link_indices"],
    )


def set_environment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_env(use_estimated_state, seed, placement_name):
    def initialize():
        set_environment_seed(seed)
        os.chdir(REFERENCE_REPO)

        node_indices, link_indices = get_sensor_indices(placement_name)

        n_sensors = 0 if node_indices is None else len(node_indices)

        scenario_config, _ = create_network_scenario(
            network_config=NETWORK,
            n_sensors=n_sensors,
            uncertainty_seed=seed,
        )

        env = ChlorineControlEnv(
            scenario_config=scenario_config,
            network_config=NETWORK,
            n_sensors=n_sensors,
            use_estimated_state=use_estimated_state,
            chlorine_penalty_weight=CHLORINE_PENALTY_WEIGHT,
            sensor_node_indices=node_indices,
            sensor_link_indices=link_indices,
        )

        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        return Monitor(env)

    return initialize


def create_training_env(use_estimated_state, seed, placement_name):
    base_env = DummyVecEnv(
        [make_env(use_estimated_state, seed, placement_name)]
    )

    return VecNormalize(
        base_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )


def create_validation_env(use_estimated_state, seed, placement_name):
    base_env = DummyVecEnv(
        [make_env(use_estimated_state, seed, placement_name)]
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


def get_last_model_paths(variant_name, training_seed):
    model_name = (
        f"{NETWORK_NAME}_ppo_{variant_name}"
        f"_seed{training_seed}_{RUN_NAME}"
    )

    model_path = MODEL_DIR / model_name
    normalization_path = MODEL_DIR / f"{model_name}_vecnormalize.pkl"

    return model_path, normalization_path


def get_best_model_paths(variant_name, training_seed):
    model_name = (
        f"{NETWORK_NAME}_ppo_{variant_name}"
        f"_seed{training_seed}_{RUN_NAME}"
    )

    save_dir = MODEL_DIR / "best_models" / model_name

    return (
        save_dir,
        save_dir / "best_model",
        save_dir / "best_vecnormalize.pkl",
    )


class ValidationCallback(BaseCallback):
    def __init__(
        self,
        validation_env,
        validation_seed,
        eval_freq,
        eval_steps,
        save_dir,
        chlorine_penalty_weight,
        verbose=1,
    ):
        super().__init__(verbose=verbose)

        self.validation_env = validation_env
        self.validation_seed = validation_seed
        self.eval_freq = eval_freq
        self.eval_steps = eval_steps
        self.save_dir = save_dir
        self.chlorine_penalty_weight = chlorine_penalty_weight

        self.best_score = np.inf
        self.validation_number = 0
        self.log_path = save_dir / "validation_log.csv"

    def _init_callback(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        with self.log_path.open("w", newline="", encoding="utf-8") as file:
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
                    "is_new_best",
                ]
            )

    def _run_validation(self):
        sync_envs_normalization(
            self.training_env,
            self.validation_env,
        )

        self.validation_env.training = False
        self.validation_env.norm_reward = False
        self.validation_env.seed(self.validation_seed)

        observation = self.validation_env.reset()

        rewards = []
        actions = []

        total_node_values = 0
        lower_violation_count = 0
        upper_violation_count = 0
        total_violation = 0.0

        for _ in range(self.eval_steps):
            action, _ = self.model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, done, infos = (
                self.validation_env.step(action)
            )

            info = infos[0]

            reward_value = float(np.asarray(reward).reshape(-1)[0])

            action_value = float(
                info.get(
                    "physical_chlorine_action",
                    np.asarray(action).reshape(-1)[0],
                )
            )

            if "scada_data" not in info:
                raise RuntimeError(
                    "Validation info does not contain 'scada_data'."
                )

            node_quality = np.asarray(
                info["scada_data"].get_data_nodes_quality(),
                dtype=float,
            ).reshape(-1)

            if not np.all(np.isfinite(node_quality)):
                raise RuntimeError(
                    "Validation produced non-finite chlorine concentrations."
                )

            lower_violation = np.maximum(
                LOWER_CHLORINE_BOUND - node_quality,
                0.0,
            )

            upper_violation = np.maximum(
                node_quality - UPPER_CHLORINE_BOUND,
                0.0,
            )

            lower_violation_count += int(
                np.sum(node_quality < LOWER_CHLORINE_BOUND)
            )

            upper_violation_count += int(
                np.sum(node_quality > UPPER_CHLORINE_BOUND)
            )

            total_violation += float(
                np.sum(lower_violation + upper_violation)
            )

            total_node_values += node_quality.size
            rewards.append(reward_value)
            actions.append(action_value)

            if bool(done[0]):
                self.validation_env.seed(self.validation_seed)
                observation = self.validation_env.reset()

        mean_action = float(np.mean(actions))

        score = (
            total_violation / self.eval_steps
            + self.chlorine_penalty_weight * mean_action
        )

        return {
            "score": float(score),
            "mean_reward": float(np.mean(rewards)),
            "mean_action": mean_action,
            "cumulative_action": float(np.sum(actions)),
            "outside_range_fraction": (
                lower_violation_count + upper_violation_count
            ) / total_node_values,
            "lower_violation_fraction": (
                lower_violation_count / total_node_values
            ),
            "upper_violation_fraction": (
                upper_violation_count / total_node_values
            ),
            "mean_violation_per_node_value": (
                total_violation / total_node_values
            ),
        }

    def _save_best_model(self, metrics):
        model_path = self.save_dir / "best_model"
        normalization_path = self.save_dir / "best_vecnormalize.pkl"
        metrics_path = self.save_dir / "best_validation_metrics.json"

        self.model.save(str(model_path))

        normalization_env = self.model.get_vec_normalize_env()

        if normalization_env is None:
            raise RuntimeError(
                "Training environment does not contain VecNormalize."
            )

        normalization_env.save(str(normalization_path))

        output = {
            "training_timesteps": int(self.num_timesteps),
            **metrics,
        }

        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2)

    def _append_log(self, metrics, is_new_best):
        with self.log_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    self.validation_number,
                    self.num_timesteps,
                    metrics["score"],
                    metrics["mean_reward"],
                    metrics["mean_action"],
                    metrics["cumulative_action"],
                    metrics["outside_range_fraction"],
                    metrics["lower_violation_fraction"],
                    metrics["upper_violation_fraction"],
                    metrics["mean_violation_per_node_value"],
                    is_new_best,
                ]
            )

    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True

        self.validation_number += 1
        metrics = self._run_validation()

        is_new_best = metrics["score"] < self.best_score

        if is_new_best:
            self.best_score = metrics["score"]
            self._save_best_model(metrics)

        self._append_log(metrics, is_new_best)

        self.logger.record("validation/score", metrics["score"])

        if self.verbose:
            status = "new best" if is_new_best else "not improved"

            print(
                f"Validation at {self.num_timesteps} steps: "
                f"score={metrics['score']:.4f}, "
                f"outside={metrics['outside_range_fraction']:.4f}, "
                f"action={metrics['mean_action']:.4f}, "
                f"{status}"
            )

        return True


def train_variant(
    variant_name,
    use_estimated_state,
    placement_name,
    training_seed,
):
    last_model_path, last_normalization_path = get_last_model_paths(
        variant_name,
        training_seed,
    )

    (
        best_model_dir,
        best_model_path,
        best_normalization_path,
    ) = get_best_model_paths(
        variant_name,
        training_seed,
    )

    last_model_zip = Path(f"{last_model_path}.zip")
    best_model_zip = Path(f"{best_model_path}.zip")

    all_files_exist = (
        last_model_zip.exists()
        and last_normalization_path.exists()
        and best_model_zip.exists()
        and best_normalization_path.exists()
    )

    if SKIP_EXISTING_MODELS and all_files_exist:
        print(f"Skipping {variant_name}, seed {training_seed}")
        return best_model_path, best_normalization_path

    print(f"Training {variant_name}, seed {training_seed}")

    env = create_training_env(
        use_estimated_state,
        training_seed,
        placement_name,
    )

    validation_env = create_validation_env(
        use_estimated_state,
        VALIDATION_SEED,
        placement_name,
    )

    callback = ValidationCallback(
        validation_env=validation_env,
        validation_seed=VALIDATION_SEED,
        eval_freq=VALIDATION_FREQ,
        eval_steps=VALIDATION_STEPS,
        save_dir=best_model_dir,
        chlorine_penalty_weight=CHLORINE_PENALTY_WEIGHT,
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

        model.save(str(last_model_path))
        env.save(str(last_normalization_path))

    finally:
        env.close()
        validation_env.close()

    if not best_model_zip.exists():
        raise RuntimeError(
            "No best model was saved. Check the validation settings."
        )

    if not best_normalization_path.exists():
        raise RuntimeError(
            "No VecNormalize file was saved for the best model."
        )

    return best_model_path, best_normalization_path


def evaluate_variant(
    variant_name,
    use_estimated_state,
    placement_name,
    training_seed,
    model_path,
    normalization_path,
):
    set_environment_seed(EVALUATION_SEED)

    base_env = DummyVecEnv(
        [
            make_env(
                use_estimated_state,
                EVALUATION_SEED,
                placement_name,
            )
        ]
    )

    env = VecNormalize.load(
        str(normalization_path),
        base_env,
    )

    env.training = False
    env.norm_reward = False
    env.seed(EVALUATION_SEED)

    model = PPO.load(
        str(model_path),
        env=env,
        device="cpu",
    )

    rewards = []
    actions = []

    total_node_values = 0
    lower_violation_count = 0
    upper_violation_count = 0
    total_violation = 0.0

    node_lower_counts = np.zeros(NETWORK.n_nodes, dtype=int)
    node_upper_counts = np.zeros(NETWORK.n_nodes, dtype=int)
    node_total_violation = np.zeros(NETWORK.n_nodes, dtype=float)

    selected_training_timesteps = -1

    metrics_path = model_path.parent / "best_validation_metrics.json"

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as file:
            validation_metrics = json.load(file)

        selected_training_timesteps = int(
            validation_metrics["training_timesteps"]
        )

    print(
        f"Evaluating {variant_name}, seed {training_seed}, "
        f"checkpoint {selected_training_timesteps}"
    )

    try:
        observation = env.reset()

        for _ in range(EVALUATION_STEPS):
            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, done, infos = env.step(action)
            info = infos[0]

            reward_value = float(np.asarray(reward).reshape(-1)[0])

            action_value = float(
                info.get(
                    "physical_chlorine_action",
                    np.asarray(action).reshape(-1)[0],
                )
            )

            if "scada_data" not in info:
                raise RuntimeError(
                    "Evaluation info does not contain 'scada_data'."
                )

            node_quality = np.asarray(
                info["scada_data"].get_data_nodes_quality(),
                dtype=float,
            ).reshape(-1)

            if not np.all(np.isfinite(node_quality)):
                raise RuntimeError(
                    "Evaluation produced non-finite chlorine concentrations."
                )

            lower_mask = node_quality < LOWER_CHLORINE_BOUND
            upper_mask = node_quality > UPPER_CHLORINE_BOUND

            lower_violation = np.maximum(
                LOWER_CHLORINE_BOUND - node_quality,
                0.0,
            )

            upper_violation = np.maximum(
                node_quality - UPPER_CHLORINE_BOUND,
                0.0,
            )

            violation = lower_violation + upper_violation

            lower_violation_count += int(np.sum(lower_mask))
            upper_violation_count += int(np.sum(upper_mask))
            total_violation += float(np.sum(violation))
            total_node_values += node_quality.size

            node_lower_counts += lower_mask.astype(int)
            node_upper_counts += upper_mask.astype(int)
            node_total_violation += violation

            rewards.append(reward_value)
            actions.append(action_value)

            if bool(done[0]):
                env.seed(EVALUATION_SEED)
                observation = env.reset()

        result = {
            "variant": variant_name,
            "training_seed": training_seed,
            "validation_seed": VALIDATION_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "selected_training_timesteps": selected_training_timesteps,
            "outside_range_fraction": (
                lower_violation_count + upper_violation_count
            ) / total_node_values,
            "lower_violation_fraction": (
                lower_violation_count / total_node_values
            ),
            "upper_violation_fraction": (
                upper_violation_count / total_node_values
            ),
            "mean_violation_per_node_value": (
                total_violation / total_node_values
            ),
            "cumulative_action": float(np.sum(actions)),
            "mean_reward": float(np.mean(rewards)),
        }

        spatial_results = []

        for node_index in range(NETWORK.n_nodes):
            spatial_results.append(
                {
                    "variant": variant_name,
                    "training_seed": training_seed,
                    "node_index": node_index,
                    "outside_range_fraction": (
                        node_lower_counts[node_index]
                        + node_upper_counts[node_index]
                    ) / EVALUATION_STEPS,
                    "lower_violation_fraction": (
                        node_lower_counts[node_index]
                        / EVALUATION_STEPS
                    ),
                    "upper_violation_fraction": (
                        node_upper_counts[node_index]
                        / EVALUATION_STEPS
                    ),
                    "mean_violation": (
                        node_total_violation[node_index]
                        / EVALUATION_STEPS
                    ),
                }
            )

        print(
            f"Result: outside={result['outside_range_fraction']:.4f}, "
            f"violation={result['mean_violation_per_node_value']:.4f}, "
            f"chlorine={result['cumulative_action']:.2f}"
        )

        return result, spatial_results

    finally:
        env.close()


def save_csv(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def calculate_std(values):
    if len(values) < 2:
        return float("nan")

    return float(np.std(values, ddof=1))


def save_summary(results):
    rows = []

    for variant_name in VARIANTS:
        variant_results = [
            result
            for result in results
            if result["variant"] == variant_name
        ]

        row = {
            "variant": variant_name,
            "n_training_seeds": len(variant_results),
        }

        for metric in SUMMARY_METRICS:
            values = np.asarray(
                [result[metric] for result in variant_results],
                dtype=float,
            )

            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = calculate_std(values)

        rows.append(row)

    output_path = (
        RESULTS_DIR
        / f"{NETWORK_NAME}_ppo_{RUN_NAME}_summary.csv"
    )

    save_csv(rows, output_path)

    return output_path


def print_summary(results):
    print("\nSummary")

    for variant_name in VARIANTS:
        variant_results = [
            result
            for result in results
            if result["variant"] == variant_name
        ]

        print(f"\n{variant_name}")

        for metric in SUMMARY_METRICS:
            values = np.asarray(
                [result[metric] for result in variant_results],
                dtype=float,
            )

            mean = float(np.mean(values))
            std = calculate_std(values)

            print(f"{metric}: {mean:.4f} ± {std:.4f}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    spatial_results = []

    for variant_name, variant_config in VARIANTS.items():
        for training_seed in TRAINING_SEEDS:
            model_path, normalization_path = train_variant(
                variant_name=variant_name,
                use_estimated_state=variant_config["use_estimated_state"],
                placement_name=variant_config["placement"],
                training_seed=training_seed,
            )

            result, node_results = evaluate_variant(
                variant_name=variant_name,
                use_estimated_state=variant_config["use_estimated_state"],
                placement_name=variant_config["placement"],
                training_seed=training_seed,
                model_path=model_path,
                normalization_path=normalization_path,
            )

            results.append(result)
            spatial_results.extend(node_results)

            raw_path = (
                RESULTS_DIR
                / f"{NETWORK_NAME}_ppo_{RUN_NAME}_raw.csv"
            )

            spatial_path = (
                RESULTS_DIR
                / f"{NETWORK_NAME}_ppo_{RUN_NAME}_spatial.csv"
            )

            save_csv(results, raw_path)
            save_csv(spatial_results, spatial_path)

    summary_path = save_summary(results)

    print_summary(results)

    print(f"\nSaved results to: {RESULTS_DIR}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()