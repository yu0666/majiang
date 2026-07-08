# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# class DecisionVisualizer:
#     def __init__(self, save_dir="xai_visualizations1"):
#         self.save_dir = save_dir
#         os.makedirs(self.save_dir, exist_ok=True)

#     def plot_action_confidence(self, step: int, action_probs: dict):
#         plt.figure(figsize=(8, 5))
#         actions = list(action_probs.keys())
#         probs = list(action_probs.values())
        
#         sns.barplot(x=probs, y=actions, hue=actions, palette="viridis", legend=False)
        
#         plt.xlim(0, 1.0)
#         plt.title(f"Step {step}: Agent Action Confidence", fontsize=14, pad=15)
#         plt.xlabel("Probability", fontsize=12)
#         plt.ylabel("Candidate Actions", fontsize=12)
        
#         for i, v in enumerate(probs):
#             plt.text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=11, fontweight='bold')
            
#         plt.tight_layout()
#         save_path = os.path.join(self.save_dir, f"confidence_step_{step}.png")
#         plt.savefig(save_path, dpi=150)
#         plt.close()

#     def plot_attention_heatmap(self, step: int, tokens: list, attention_weights: np.ndarray, top_k=20):
#         assert len(tokens) == len(attention_weights)
        
#         top_indices = np.argsort(attention_weights)[-top_k:][::-1]
#         top_tokens = [tokens[i].replace("Ġ", "").replace(" ", "") for i in top_indices]
#         top_weights = attention_weights[top_indices]

#         plt.figure(figsize=(10, 6))
#         heatmap_data = np.array([top_weights])
        
#         sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", 
#                     xticklabels=top_tokens, yticklabels=["Attention"],
#                     cbar_kws={'label': 'Attention Weight'})
        
#         plt.title(f"Step {step}: Agent Attention Map (Top {top_k} Focus)", fontsize=14, pad=15)
#         plt.xticks(rotation=45, ha='right', fontsize=11)
#         plt.tight_layout()
        
#         save_path = os.path.join(self.save_dir, f"attention_step_{step}.png")
#         plt.savefig(save_path, dpi=150)
#         plt.close()
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

class DecisionVisualizer:
    def __init__(self, save_dir="xai_visualizations"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_action_confidence(self, step: int, action_probs: dict):
        plt.figure(figsize=(8, 5))
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        
        sns.barplot(x=probs, y=actions, hue=actions, palette="viridis", legend=False)
        
        plt.xlim(0, 1.0)
        plt.title(f"Step {step}: Agent Action Confidence", fontsize=14, pad=15)
        plt.xlabel("Probability", fontsize=12)
        plt.ylabel("Candidate Actions", fontsize=12)
        
        for i, v in enumerate(probs):
            plt.text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=11, fontweight='bold')
            
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"confidence_step_{step}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

    def plot_attention_heatmap(self, step: int, tokens: list, attention_weights: np.ndarray, top_k=20):
        assert len(tokens) == len(attention_weights)
        
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        top_tokens = [tokens[i] for i in top_indices]
        top_weights = attention_weights[top_indices]

        plt.figure(figsize=(12, 6))
        heatmap_data = np.array([top_weights])
        
        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", 
                    xticklabels=top_tokens, yticklabels=["Attention"],
                    cbar_kws={'label': 'Attention Weight'})
        
        plt.title(f"Step {step}: Agent Attention Map (Top {top_k} Focus)", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=13)
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, f"attention_step_{step}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()