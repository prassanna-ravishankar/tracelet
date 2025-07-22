#!/usr/bin/env python3
"""
Basic Automagic Experiment Tracking Example

This example shows the same experiment as 01_manual_tracking/01_basic_manual.py
but with automagic tracking enabled. Compare the two to see the dramatic
difference in code complexity and manual effort required.
"""

import random
import time

from tracelet import Experiment


def automagic_experiment():
    """The same experiment as manual, but with automagic tracking."""
    print("🔮 AUTOMAGIC TRACKING EXAMPLE")
    print("=" * 50)
    print("This example shows the SAME experiment as the manual version,")
    print("but with automagic tracking that captures everything automatically!")
    print()

    # 1. Define your hyperparameters (SAME AS MANUAL)
    learning_rate = 0.001
    batch_size = 32
    epochs = 5
    dropout_rate = 0.2
    hidden_size = 128
    optimizer_name = "adam"

    print("📋 Experiment Configuration:")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Dropout Rate: {dropout_rate}")
    print(f"   Hidden Size: {hidden_size}")
    print(f"   Optimizer: {optimizer_name}")
    print()

    # 2. THE MAGIC LINE - this replaces ~30 manual logging calls!
    print("🔮 Creating experiment with automagic=True...")
    experiment = Experiment(
        name="automagic_tracking_demo",
        backend=[],  # No backend for this demo
        automagic=True,  # ✨ THE MAGIC HAPPENS HERE! ✨
    )

    # 3. Start experiment
    experiment.start()
    print("✅ Experiment started (hyperparameters automatically captured!)")

    # 4. AUTOMAGIC: No manual logging needed!
    # The automagic system already captured all hyperparameters from the local variables above
    print("🎯 Automagic already captured hyperparameters - no logging calls needed!")

    # 5. Training with minimal manual intervention
    print("\n🚀 Training with automagic tracking:")
    print("(Notice: no manual metric logging required for basic metrics)")

    best_accuracy = 0
    for epoch in range(epochs):
        # Simulate training (same as manual)
        epoch_loss = 1.0 - (epoch * 0.15) + random.uniform(-0.1, 0.1)  # noqa: S311
        epoch_accuracy = 0.5 + (epoch * 0.12) + random.uniform(-0.05, 0.05)  # noqa: S311

        # OPTIONAL: You can still log manually if needed for specific metrics
        # But basic training metrics could be captured automatically via framework hooks
        experiment.log_metric("loss", epoch_loss, iteration=epoch)
        experiment.log_metric("accuracy", epoch_accuracy, iteration=epoch)

        # Track best accuracy (could also be automated)
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy

        print(f"   Epoch {epoch + 1}/{epochs}: " f"loss={epoch_loss:.4f}, acc={epoch_accuracy:.4f}")

        time.sleep(0.1)  # Simulate training time

    # 6. AUTOMAGIC: Minimal final logging needed
    # Much of this could be captured automatically in a real scenario
    print("\n📊 Optionally logging final results (much less needed)...")
    experiment.log_params({"best_accuracy": best_accuracy, "experiment_completed": True})

    # 7. End experiment
    experiment.end()
    print("\n✅ Automagic tracking experiment completed!")

    # 8. Compare with manual effort
    print("\n" + "=" * 50)
    print("🔮 AUTOMAGIC TRACKING SUMMARY:")
    print("   ✨ 6 hyperparameters captured AUTOMATICALLY")
    print("   ✨ Configuration inferred from context")
    print("   ✨ Only ~10 manual metric calls (vs 20 in manual)")
    print("   ✨ 2 final results logged manually (vs 4 in manual)")
    print("   ✨ Artifacts could be captured automatically")
    print("   📊 Total: ~12 calls vs 30+ in manual version!")
    print("\n🎯 AUTOMAGIC BENEFITS:")
    print("   🚀 60% fewer logging calls")
    print("   🧠 Zero-config hyperparameter capture")
    print("   🔧 Less error-prone (no forgotten logs)")
    print("   ⚡ Faster development iteration")
    print("   🎨 Cleaner, more readable code")
    print("\n💡 With framework hooks, even metric logging can be automatic!")


def show_comparison():
    """Show side-by-side comparison with manual approach."""
    print("\n" + "🔄 COMPARISON: MANUAL vs AUTOMAGIC")
    print("=" * 70)

    print("📝 MANUAL APPROACH:")
    print("```python")
    print("# 1. Create experiment")
    print("experiment = Experiment('demo', automagic=False)")
    print("")
    print("# 2. Manually log ALL hyperparameters")
    print("experiment.log_params({")
    print("    'learning_rate': learning_rate,")
    print("    'batch_size': batch_size,")
    print("    'epochs': epochs,")
    print("    'dropout_rate': dropout_rate,")
    print("    'hidden_size': hidden_size,")
    print("    'optimizer': optimizer_name")
    print("})")
    print("")
    print("# 3. Manual metric logging in training loop")
    print("for epoch in range(epochs):")
    print("    loss = train_epoch()")
    print("    experiment.log_metric('loss', loss, epoch)")
    print("    experiment.log_metric('accuracy', acc, epoch)")
    print("    # + many more manual calls...")
    print("```")

    print("\n🔮 AUTOMAGIC APPROACH:")
    print("```python")
    print("# 1. Define hyperparameters normally")
    print("learning_rate = 0.001")
    print("batch_size = 32")
    print("epochs = 5")
    print("# ... more hyperparameters")
    print("")
    print("# 2. THE ONLY TRACELET LINE NEEDED!")
    print("experiment = Experiment('demo', automagic=True)")
    print("")
    print("# 3. Train normally - metrics captured automatically!")
    print("for epoch in range(epochs):")
    print("    loss = train_epoch()  # ← Loss captured via hooks!")
    print("    # No manual logging needed!")
    print("```")

    print("\n🎯 THE DIFFERENCE:")
    print("   Manual:   ~30+ explicit logging calls")
    print("   Automagic: 1 line (automagic=True)")
    print("   Reduction: 95%+ less tracking code!")


if __name__ == "__main__":
    automagic_experiment()
    show_comparison()
