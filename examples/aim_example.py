from torch.utils.tensorboard import SummaryWriter

import tracelet

# Start experiment tracking with your preferred backend
tracelet.start_logging(
    exp_name="my_aim_experiment",
    project="my_aim_project",
    backend="aim",  # Use the AIM backend
)

# Use TensorBoard as usual - metrics are automatically captured
writer = SummaryWriter()
for epoch in range(10):
    loss = 0.9**epoch  # Example loss
    writer.add_scalar("Loss/train", loss, epoch)
    # Metrics are automatically sent to AIM!

# Log some parameters
tracelet.get_active_experiment().log_params({"learning_rate": 0.001, "batch_size": 32, "epochs": 10})

# Log an artifact (e.g., a dummy model file)
with open("dummy_model.txt", "w") as f:
    f.write("This is a dummy model file.")
tracelet.get_active_experiment().log_artifact("dummy_model.txt")

# Stop tracking when done
tracelet.stop_logging()
