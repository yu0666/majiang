# import os

# import matplotlib.pyplot as plt
# import numpy as np


# RESULTS_DIR = "eval_results"
# MODEL_TAGS = ["qwen", "qwen-sft-bot", "qwen-sft-selfplay", "qwen-grpo"]
# MODEL_DISPLAY_NAMES = {
#     "qwen": "Qwen",
#     "qwen-sft-bot": "Qwen-SFT-Bot",
#     "qwen-sft-selfplay": "Qwen-SFT-SelfPlay",
#     "qwen-grpo": "Qwen-GRPO",
# }
# MODEL_COLORS = {
#     "qwen": "#666666",
#     "qwen-sft-bot": "#FFB347",
#     "qwen-sft-selfplay": "#FF8844",
#     "qwen-grpo": "#FF4444",
# }
# LLM_PLAYER_ID = 0


# def configure_matplotlib():
#     plt.rcParams["font.family"] = "DejaVu Sans"
#     plt.rcParams["axes.unicode_minus"] = False
#     plt.rcParams["figure.dpi"] = 150


# def load_saved_summary(model_tag):
#     path = os.path.join(RESULTS_DIR, f"{model_tag}_summary.npz")
#     if not os.path.exists(path):
#         return None

#     data = np.load(path, allow_pickle=True)
#     return {
#         "model_tag": model_tag,
#         "avg_stats": {
#             "hu_count": np.array(data["hu_count"], dtype=float),
#             "dianpao_count": np.array(data["dianpao_count"], dtype=float),
#             "total_fan": np.array(data["total_fan"], dtype=float),
#         },
#         "num_episodes": int(data["num_episodes"]),
#     }


# def build_metric_data(model_tags):
#     summaries = []
#     missing_tags = []

#     for tag in model_tags:
#         summary = load_saved_summary(tag)
#         if summary is None:
#             missing_tags.append(tag)
#         else:
#             summaries.append(summary)

#     if missing_tags:
#         raise FileNotFoundError(f"Missing result files: {', '.join(missing_tags)}")

#     labels = [MODEL_DISPLAY_NAMES.get(s["model_tag"], s["model_tag"]) for s in summaries]
#     colors = [MODEL_COLORS.get(s["model_tag"], "#333333") for s in summaries]

#     hu_rates = []
#     dianpao_rates = []
#     total_fans = []

#     for summary in summaries:
#         num_episodes = summary["num_episodes"]
#         hu_count = float(summary["avg_stats"]["hu_count"][LLM_PLAYER_ID])
#         dianpao_count = float(summary["avg_stats"]["dianpao_count"][LLM_PLAYER_ID])
#         total_fan = float(summary["avg_stats"]["total_fan"][LLM_PLAYER_ID])

#         hu_rates.append((hu_count / num_episodes) * 100 if num_episodes > 0 else 0.0)
#         dianpao_rates.append((dianpao_count / num_episodes) * 100 if num_episodes > 0 else 0.0)
#         total_fans.append(total_fan)

#     return labels, colors, hu_rates, dianpao_rates, total_fans


# def add_bar_labels(ax, bars, values, fmt):
#     upper = max(values) if values else 0.0
#     offset = upper * 0.03 if upper > 0 else 0.03

#     for bar, value in zip(bars, values):
#         ax.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height() + offset,
#             fmt.format(value),
#             ha="center",
#             va="bottom",
#             fontsize=10,
#             fontweight="bold",
#         )


# def plot_metric_bar_chart(model_tags=None):
#     if model_tags is None:
#         model_tags = MODEL_TAGS

#     configure_matplotlib()
#     labels, colors, hu_rates, dianpao_rates, total_fans = build_metric_data(model_tags)
#     x = np.arange(len(labels))

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     metric_specs = [
#         ("Win Rate", "Rate (%)", hu_rates, "{:.2f}%"),
#         ("Discard Loss Rate", "Rate (%)", dianpao_rates, "{:.2f}%"),
#         ("Total Fan", "Fan Count", total_fans, "{:.2f}"),
#     ]
#     figure_title = "Core Metrics Comparison of Four Models"

#     for ax, (title, ylabel, values, fmt) in zip(axes, metric_specs):
#         bars = ax.bar(x, values, color=colors, width=0.62)
#         ax.set_xticks(x)
#         ax.set_xticklabels(labels, rotation=12)
#         ax.set_title(title, fontsize=13, fontweight="bold")
#         ax.set_ylabel(ylabel)
#         ax.grid(axis="y", alpha=0.3, linestyle="--")

#         upper = max(values) if values else 0.0
#         ax.set_ylim(0, upper * 1.18 if upper > 0 else 1.0)
#         add_bar_labels(ax, bars, values, fmt)

#     fig.suptitle(figure_title, fontsize=16, fontweight="bold")
#     plt.tight_layout()

#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     output_path = os.path.join(RESULTS_DIR, "model_metrics_bar.png")
#     plt.savefig(output_path, dpi=150, bbox_inches="tight")
#     plt.close()
#     print(f"Saved chart to: {output_path}")


# if __name__ == "__main__":
#     plot_metric_bar_chart()

import os

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "eval_results"
MODEL_TAGS = ["qwen", "qwen-sft-bot", "qwen-sft-selfplay", "qwen-grpo"]
INCLUDE_BOT_AVG = True
MODEL_DISPLAY_NAMES = {
    "qwen": "Qwen",
    "qwen-sft-bot": "Qwen-SFT-Bot",
    "qwen-sft-selfplay": "Qwen-SFT-SelfPlay",
    "qwen-grpo": "Qwen-GRPO",
    "bot-avg": "Bot Avg",
}
MODEL_COLORS = {
    "qwen": "#666666",
    "qwen-sft-bot": "#FFB347",
    "qwen-sft-selfplay": "#FF8844",
    "qwen-grpo": "#FF4444",
    "bot-avg": "#2A7FFF",
}
LLM_PLAYER_ID = 0
BOT_PLAYER_IDS = [1, 2, 3]


def configure_matplotlib():
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def load_saved_summary(model_tag):
    path = os.path.join(RESULTS_DIR, f"{model_tag}_summary.npz")
    if not os.path.exists(path):
        return None

    data = np.load(path, allow_pickle=True)
    return {
        "model_tag": model_tag,
        "avg_stats": {
            "hu_count": np.array(data["hu_count"], dtype=float),
            "dianpao_count": np.array(data["dianpao_count"], dtype=float),
            "total_fan": np.array(data["total_fan"], dtype=float),
        },
        "num_episodes": int(data["num_episodes"]),
    }


def build_metric_data(model_tags):
    summaries = []
    missing_tags = []

    for tag in model_tags:
        summary = load_saved_summary(tag)
        if summary is None:
            missing_tags.append(tag)
        else:
            summaries.append(summary)

    if missing_tags:
        raise FileNotFoundError(f"Missing result files: {', '.join(missing_tags)}")

    labels = [MODEL_DISPLAY_NAMES.get(s["model_tag"], s["model_tag"]) for s in summaries]
    colors = [MODEL_COLORS.get(s["model_tag"], "#333333") for s in summaries]

    hu_rates = []
    dianpao_rates = []
    total_fans = []

    for summary in summaries:
        num_episodes = summary["num_episodes"]
        hu_count = float(summary["avg_stats"]["hu_count"][LLM_PLAYER_ID])
        dianpao_count = float(summary["avg_stats"]["dianpao_count"][LLM_PLAYER_ID])
        total_fan = float(summary["avg_stats"]["total_fan"][LLM_PLAYER_ID])

        hu_rates.append((hu_count / num_episodes) * 100 if num_episodes > 0 else 0.0)
        dianpao_rates.append((dianpao_count / num_episodes) * 100 if num_episodes > 0 else 0.0)
        total_fans.append(total_fan)

    if INCLUDE_BOT_AVG:
        bot_hu_rates = []
        bot_dianpao_rates = []
        bot_total_fans = []

        for summary in summaries:
            num_episodes = summary["num_episodes"]
            bot_hu_count = np.mean([float(summary["avg_stats"]["hu_count"][i]) for i in BOT_PLAYER_IDS])
            bot_dianpao_count = np.mean([float(summary["avg_stats"]["dianpao_count"][i]) for i in BOT_PLAYER_IDS])
            bot_total_fan = np.mean([float(summary["avg_stats"]["total_fan"][i]) for i in BOT_PLAYER_IDS])

            bot_hu_rates.append((bot_hu_count / num_episodes) * 100 if num_episodes > 0 else 0.0)
            bot_dianpao_rates.append((bot_dianpao_count / num_episodes) * 100 if num_episodes > 0 else 0.0)
            bot_total_fans.append(bot_total_fan)

        labels.append(MODEL_DISPLAY_NAMES["bot-avg"])
        colors.append(MODEL_COLORS["bot-avg"])
        hu_rates.append(float(np.mean(bot_hu_rates)))
        dianpao_rates.append(float(np.mean(bot_dianpao_rates)))
        total_fans.append(float(np.mean(bot_total_fans)))

    return labels, colors, hu_rates, dianpao_rates, total_fans


def add_bar_labels(ax, bars, values, fmt):
    upper = max(values) if values else 0.0
    offset = upper * 0.03 if upper > 0 else 0.03

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def plot_metric_bar_chart(model_tags=None):
    if model_tags is None:
        model_tags = MODEL_TAGS

    configure_matplotlib()
    labels, colors, hu_rates, dianpao_rates, total_fans = build_metric_data(model_tags)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metric_specs = [
        ("Win Rate", "Rate (%)", hu_rates, "{:.2f}%"),
        ("Discard Loss Rate", "Rate (%)", dianpao_rates, "{:.2f}%"),
        ("Total Fan", "Fan Count", total_fans, "{:.2f}"),
    ]
    figure_title = "Core Metrics Comparison of Models and Bot Baseline"

    for ax, (title, ylabel, values, fmt) in zip(axes, metric_specs):
        bars = ax.bar(x, values, color=colors, width=0.62)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        upper = max(values) if values else 0.0
        ax.set_ylim(0, upper * 1.18 if upper > 0 else 1.0)
        add_bar_labels(ax, bars, values, fmt)

    fig.suptitle(figure_title, fontsize=16, fontweight="bold")
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(RESULTS_DIR, "model_metrics_bar.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart to: {output_path}")


if __name__ == "__main__":
    plot_metric_bar_chart()
