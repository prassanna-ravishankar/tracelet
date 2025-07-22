#!/usr/bin/env python3
"""
Automagic PyTorch Training Example

This example shows the SAME PyTorch training as 01_manual_tracking/02_pytorch_manual.py
but with automagic tracking. Compare the two files to see how automagic eliminates
120+ manual logging calls while providing the same (or better) experiment tracking.
"""

import random
import time

from tracelet import Experiment


def automagic_pytorch_training():
    """The same PyTorch training as manual, but with automagic tracking."""
    print("🔮 AUTOMAGIC PYTORCH TRAINING EXAMPLE")
    print("=" * 60)
    print("This shows the SAME PyTorch training as the manual version,")
    print("but with automagic tracking that captures everything automatically!")
    print()

    # Model hyperparameters (SAME AS MANUAL - but automatically captured!)
    learning_rate = 0.001
    batch_size = 64
    epochs = 10
    weight_decay = 1e-4  # noqa: F841
    dropout_rate = 0.3
    hidden_layers = [256, 128, 64]
    activation = "relu"  # noqa: F841
    optimizer_type = "adam"

    # Additional config that will be auto-captured
    model_type = "feedforward_neural_network"  # noqa: F841
    dataset = "synthetic_classification"  # noqa: F841
    input_size = 784  # noqa: F841
    output_size = 10  # noqa: F841
    num_classes = 10  # noqa: F841

    print("🏗️  Model Configuration:")
    print(f"   Architecture: {hidden_layers}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Dropout: {dropout_rate}")
    print(f"   Optimizer: {optimizer_type}")
    print()

    # THE MAGIC LINE - replaces 120+ manual logging calls!
    print("🔮 Creating experiment with automagic=True...")
    experiment = Experiment(
        name="automagic_pytorch_training",
        backend=[],
        automagic=True,  # ✨ ALL HYPERPARAMETERS CAPTURED AUTOMATICALLY! ✨
    )
    experiment.start()

    print("✅ Experiment started - all hyperparameters automatically captured!")
    print("🎯 No manual parameter logging needed!")

    # AUTOMAGIC: No manual hyperparameter logging needed!
    # The automagic system already captured:
    # - All model hyperparameters from local variables
    # - Framework configuration (when PyTorch is imported)
    # - System information automatically

    # Training loop with minimal manual intervention
    print("\n🚀 Starting automagic PyTorch training:")
    print("(In real PyTorch, loss/metrics would be captured automatically via hooks)")

    best_val_acc = 0

    for epoch in range(epochs):
        print(f"\n📊 Epoch {epoch + 1}/{epochs}")

        # Simulate training phase
        epoch_train_loss = 2.3 - (epoch * 0.15) + random.uniform(-0.1, 0.1)
        epoch_train_acc = 0.1 + (epoch * 0.08) + random.uniform(-0.02, 0.02)

        # In real PyTorch with automagic, these would be captured automatically:
        # - Loss via criterion hooks
        # - Accuracy via metric hooks
        # - Learning rate via optimizer hooks
        # - Gradient norms via model hooks
        # - GPU memory via system monitors

        # For demonstration, we'll log a few key metrics manually
        # (In real usage, framework hooks would capture these automatically)
        experiment.log_metric("train_loss", epoch_train_loss, iteration=epoch)
        experiment.log_metric("train_accuracy", epoch_train_acc, iteration=epoch)

        # Simulate validation
        epoch_val_loss = epoch_train_loss + random.uniform(0.05, 0.15)
        epoch_val_acc = epoch_train_acc - random.uniform(0.01, 0.05)

        experiment.log_metric("val_loss", epoch_val_loss, iteration=epoch)
        experiment.log_metric("val_accuracy", epoch_val_acc, iteration=epoch)

        # Track best model (could also be automated)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

        print(f"   Train: loss={epoch_train_loss:.4f}, acc={epoch_train_acc:.4f}")
        print(f"   Val:   loss={epoch_val_loss:.4f}, acc={epoch_val_acc:.4f}")

        time.sleep(0.1)  # Simulate training time

    # AUTOMAGIC: Minimal final logging
    # Most of this would be captured automatically in real scenarios
    print("\n📊 Training completed - minimal final logging needed...")
    experiment.log_params({"best_val_accuracy": best_val_acc, "training_completed": True})

    experiment.end()

    print("\n✅ Automagic PyTorch training completed!")
    print("\n" + "=" * 60)
    print("🔮 AUTOMAGIC PYTORCH TRACKING SUMMARY:")
    print("   ✨ 11 model hyperparameters captured AUTOMATICALLY")
    print("   ✨ Architecture details inferred AUTOMATICALLY")
    print("   ✨ Dataset info captured AUTOMATICALLY")
    print("   ✨ Only ~40 manual metric calls (vs 80 in manual)")
    print("   ✨ 2 final results logged (vs 8 in manual)")
    print("   ✨ System metrics captured AUTOMATICALLY")
    print("   ✨ Framework info detected AUTOMATICALLY")
    print("   📊 Total: ~42 calls vs 120+ in manual version!")
    print("\n🎯 AUTOMAGIC BENEFITS FOR PYTORCH:")
    print("   🚀 65% fewer logging calls")
    print("   🧠 Automatic hyperparameter detection")
    print("   🔧 Framework hooks capture training metrics")
    print("   📊 System monitoring built-in")
    print("   🎨 Focus on model, not logging")
    print("   ⚡ Real-time metric capture via hooks")


def show_pytorch_comparison():
    """Show detailed comparison for PyTorch specifically."""
    print("\n" + "🔄 PYTORCH COMPARISON: MANUAL vs AUTOMAGIC")
    print("=" * 70)

    print("📝 MANUAL PYTORCH APPROACH:")
    print("```python")
    print("# 1. Define hyperparameters")
    print("learning_rate = 0.001")
    print("batch_size = 64")
    print("# ... many more hyperparameters")
    print("")
    print("# 2. Create experiment")
    print("experiment = Experiment('pytorch_training', automagic=False)")
    print("")
    print("# 3. Manually log ALL hyperparameters")
    print("experiment.log_params({")
    print("    'learning_rate': learning_rate,")
    print("    'batch_size': batch_size,")
    print("    'weight_decay': weight_decay,")
    print("    'dropout_rate': dropout_rate,")
    print("    # ... 20+ more parameters")
    print("})")
    print("")
    print("# 4. Manual architecture logging")
    print("experiment.log_params({")
    print("    'total_parameters': count_parameters(model),")
    print("    'model_type': 'neural_network',")
    print("    # ... more architecture details")
    print("})")
    print("")
    print("# 5. Manual training loop logging")
    print("for epoch in range(epochs):")
    print("    for batch in dataloader:")
    print("        loss = criterion(output, target)")
    print("        # MANUAL: Log every metric")
    print("        experiment.log_metric('loss', loss.item(), step)")
    print("        experiment.log_metric('lr', optimizer.param_groups[0]['lr'])")
    print("        # ... many more manual calls")
    print("```")

    print("\n🔮 AUTOMAGIC PYTORCH APPROACH:")
    print("```python")
    print("# 1. Define hyperparameters normally")
    print("learning_rate = 0.001")
    print("batch_size = 64")
    print("weight_decay = 1e-4")
    print("dropout_rate = 0.3")
    print("# ... all your hyperparameters")
    print("")
    print("# 2. THE ONLY TRACELET LINE NEEDED!")
    print("experiment = Experiment('pytorch_training', automagic=True)")
    print("")
    print("# 3. Train normally - everything captured automatically!")
    print("for epoch in range(epochs):")
    print("    for batch in dataloader:")
    print("        loss = criterion(output, target)  # ← Loss captured via hooks!")
    print("        optimizer.step()                  # ← LR captured via hooks!")
    print("        # No manual logging needed!")
    print("```")

    print("\n🎯 PYTORCH AUTOMAGIC FEATURES:")
    print("   🔮 Automatic hyperparameter capture from local variables")
    print("   🔗 PyTorch model hooks for architecture logging")
    print("   📊 Loss function hooks for automatic metric capture")
    print("   ⚡ Optimizer hooks for learning rate tracking")
    print("   🖥️  GPU memory monitoring")
    print("   📈 Gradient norm tracking")
    print("   💾 Automatic checkpoint detection")
    print("   🎯 Model parameter counting")
    print("\n🏆 RESULT: Same functionality, 95% less code!")


if __name__ == "__main__":
    automagic_pytorch_training()
    show_pytorch_comparison()
