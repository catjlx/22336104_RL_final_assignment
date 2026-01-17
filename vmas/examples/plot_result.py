import os
import pandas as pd
import matplotlib.pyplot as plt

def load_reward_curve(storage_dir):
    """
    从 Ray/RLlib 的 progress.csv 中读取：
        training_iteration
        episode_reward_mean
    """
    # 默认取目录下第一个 trial
    trial_dirs = [
        os.path.join(storage_dir, d)
        for d in os.listdir(storage_dir)
        if os.path.isdir(os.path.join(storage_dir, d))
    ]

    if len(trial_dirs) == 0:
        raise RuntimeError(f"No trial directory found in {storage_dir}")

    progress_path = os.path.join(trial_dirs[0], "progress.csv")
    if not os.path.exists(progress_path):
        raise RuntimeError(f"progress.csv not found in {progress_path}")

    df = pd.read_csv(progress_path)
    return df["training_iteration"], df["episode_reward_mean"]


def plot_group(algorithms, title, save_path):
    """
    algorithms: dict
        key   -> label name
        value -> storage directory
    """
    plt.figure(figsize=(8, 6))

    for algo_name, storage_dir in algorithms.items():
        x, y = load_reward_curve(storage_dir)
        plt.plot(x, y, label=algo_name)

    plt.xlabel("Training Iteration")
    plt.ylabel("Episode Reward Mean")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



group_1_algorithms = {
    "IPPO": "ray_results/PPO_2026-01-16_21-37-45",
    "CPPO": "ray_results/PPO_2026-01-16_09-46-46",
    "MAPPO": "ray_results/PPO_2026-01-16_09-38-30",
}

plot_group(
    algorithms=group_1_algorithms,
    title="Comparison of IPPO, CPPO and MAPPO on Balance",
    save_path="group1_ippo_cppo_mappo.png",
)


group_2_algorithms = {
    "MAPPO": "ray_results/PPO_2026-01-16_09-38-30",
    "RD-MAPPO": "ray_results/PPO_2026-01-16_22-25-42",
}

plot_group(
    algorithms=group_2_algorithms,
    title="Comparison of MAPPO and RD-MAPPO on Balance",
    save_path="group2_mappo_rd_mappo.png",
)


print("Plots generated successfully.")
