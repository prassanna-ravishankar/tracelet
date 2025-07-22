# 🚀 Tracelet Quick Start Guide

Get started with Tracelet in 2 minutes! Choose your path:

## 🔮 Option 1: Automagic (Recommended - Zero Code)

```python
from tracelet import Experiment

# 1. Define your hyperparameters normally
learning_rate = 0.001
batch_size = 32
epochs = 10
dropout_rate = 0.2

# 2. THE ONLY TRACELET LINE NEEDED!
experiment = Experiment("my_experiment", automagic=True)

# 3. Train your model normally - everything tracked automatically!
for epoch in range(epochs):
    # Your training code here...
    # Loss, metrics, model info all captured automatically!
    pass

experiment.end()
```

**🎯 Result**: All hyperparameters, training metrics, and model info captured with ZERO manual logging!

## 🔧 Option 2: Manual (Full Control)

```python
from tracelet import Experiment

# 1. Create experiment
experiment = Experiment("my_experiment")
experiment.start()

# 2. Log hyperparameters manually
experiment.log_params({
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
})

# 3. Log metrics manually during training
for epoch in range(10):
    loss = train_epoch()  # Your training code
    experiment.log_metric("loss", loss, iteration=epoch)

experiment.end()
```

## 🔌 Option 3: With ML Backend

```python
from tracelet import Experiment

# Works with MLflow, Weights & Biases, ClearML, etc.
experiment = Experiment(
    name="my_experiment",
    backend=["mlflow"],  # or ["wandb"], ["clearml"]
    automagic=True
)

# Everything else stays the same!
```

## 📁 Example Files to Try

1. **Start here**: `examples/comparison_manual_vs_automagic.py` - See the difference
2. **Basic manual**: `examples/01_manual_tracking/01_basic_manual.py`
3. **Basic automagic**: `examples/02_automagic_tracking/01_basic_automagic.py`
4. **Full automagic showcase**: `examples/02_automagic_tracking/04_comprehensive_automagic.py`

## 🎯 Key Benefits

| Feature          | Manual             | Automagic        |
| ---------------- | ------------------ | ---------------- |
| Setup            | Multiple log calls | `automagic=True` |
| Hyperparameters  | Manual logging     | ✨ Automatic     |
| Training metrics | Manual logging     | ✨ Via hooks     |
| Model info       | Manual logging     | ✨ Automatic     |
| Code changes     | Many lines         | Single line      |

## 💡 Pro Tips

- 🔮 **Start with automagic** for easiest experience
- 🔧 **Add manual logging** only where needed
- 🔌 **Use backends** to integrate with your ML platform
- 📊 **Check examples/** for detailed tutorials

**Ready to eliminate experiment tracking boilerplate? Try automagic mode!** 🚀
