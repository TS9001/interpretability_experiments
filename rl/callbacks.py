"""Training callbacks for GRPO."""

from transformers import TrainerCallback


class EvenCheckpointCallback(TrainerCallback):
    """Saves N evenly-spaced checkpoints across training.

    Calculates save points at train start based on total steps,
    then triggers saves at those points.
    """

    def __init__(self, num_checkpoints: int):
        self.num_checkpoints = num_checkpoints
        self.save_steps: set[int] = set()

    def on_train_begin(self, args, state, control, **kwargs):
        if state.max_steps <= 0 or self.num_checkpoints <= 0:
            return

        # Evenly space checkpoints (excluding step 0, including last step)
        interval = state.max_steps / self.num_checkpoints
        self.save_steps = {int(round(interval * i)) for i in range(1, self.num_checkpoints + 1)}

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.save_steps:
            control.should_save = True
