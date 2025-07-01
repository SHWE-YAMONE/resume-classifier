from transformers import TrainerCallback
import matplotlib.pyplot as plt
import os

class MetricsPlotCallback(TrainerCallback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.metrics_history = {"epoch": [], "eval_loss": [], "eval_accuracy": [], "eval_f1": []}

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            epoch = state.epoch
            self.metrics_history["epoch"].append(epoch)
            self.metrics_history["eval_loss"].append(metrics.get("eval_loss"))
            self.metrics_history["eval_accuracy"].append(metrics.get("eval_accuracy"))
            self.metrics_history["eval_f1"].append(metrics.get("eval_f1"))
            self._plot_metrics()

    def _plot_metrics(self):
        os.makedirs(self.save_dir, exist_ok=True)
        for metric in ["eval_loss", "eval_accuracy", "eval_f1"]:
            plt.figure()
            plt.plot(self.metrics_history["epoch"], self.metrics_history[metric], marker='o')
            plt.title(f"{metric} over epochs")
            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.grid(True)
            plt.savefig(os.path.join(self.save_dir, f"{metric}.png"))
            plt.close()
