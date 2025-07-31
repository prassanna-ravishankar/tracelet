# Tracelet Examples & Tutorials

Welcome to Tracelet examples! These tutorials will help you get started with experiment tracking in just minutes.

## 🚀 Quick Start Path

### 1. **First Steps** (`01_manual_tracking/`)

Start here if you're new to experiment tracking or want fine-grained control.

- `01_basic_manual.py` - Your first experiment in 10 lines of code
- `02_pytorch_manual.py` - Track PyTorch training loops manually

### 2. **Automagic Tracking** (`02_automagic_tracking/`) ✨

The easiest way to add tracking - let Tracelet do the work for you!

- `01_basic_automagic.py` - See the magic of automatic metric capture
- `02_pytorch_automagic.py` - PyTorch training with zero tracking code
- `03_pytorch_lightning_automagic.py` - Lightning + automagic = ❤️
- `04_comprehensive_automagic.py` - Advanced automagic features

### 3. **Backend Integrations** (`03_backend_integrations/`)

Learn how to use specific backends or compare multiple ones.

- `wandb_integration.py` - Weights & Biases specific features
- `clearml_integration.py` - ClearML specific features
- `multi_backend_comparison.py` - Use multiple backends simultaneously
- `compare_all_backends.py` - See your metrics in W&B, MLflow, and ClearML at once!

### 4. **Advanced Features** (`04_advanced_features/`)

Ready for production? These examples show real-world usage.

- `e2e_ml_pipeline.py` - Complete ML pipeline with data versioning
- `complete_ml_pipeline.py` - Full training pipeline with all features

### 5. **PyTorch Lightning** (`05_lightning_automagic/`) ⚡

Special examples for Lightning users - the easiest integration ever!

- `simple_lightning_example.py` - Add tracking in just 3 lines!
- `train_model_with_automagic.py` - Complete Lightning training example

## 🎯 Which Example Should I Start With?

- **"I have 2 minutes"** → `05_lightning_automagic/simple_lightning_example.py`
- **"I want to understand the basics"** → `01_manual_tracking/01_basic_manual.py`
- **"I have existing PyTorch code"** → `02_automagic_tracking/02_pytorch_automagic.py`
- **"I use PyTorch Lightning"** → `05_lightning_automagic/simple_lightning_example.py`
- **"I want to compare backends"** → `03_backend_integrations/compare_all_backends.py`

## 💡 Pro Tips

1. **Automagic is magic**: If you're starting fresh, use `automagic=True`. It captures metrics automatically!

2. **Backend flexibility**: You can use one or multiple backends:

   ```python
   # Single backend
   exp = Experiment(backend=["wandb"], automagic=True)

   # Multiple backends - see metrics everywhere!
   exp = Experiment(backend=["wandb", "mlflow", "clearml"], automagic=True)
   ```

3. **Environment setup**: Create a `.env` file with your API keys:

   ```bash
   WANDB_API_KEY=your_key_here
   CLEARML_API_KEY=your_key_here
   ```

4. **Quick test**: Most examples use synthetic data so you can run them immediately without downloading datasets.

## 📚 Learn More

- [Tracelet Documentation](https://tracelet.ai/docs)
- [API Reference](https://tracelet.ai/api)
- [Discord Community](https://discord.gg/tracelet)

Happy tracking! 🎉
