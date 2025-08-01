#!/usr/bin/env python3
"""
Basic Artifact Tracking Example
===============================

This example demonstrates Tracelet's new artifact system for logging models,
checkpoints, images, and other files across different MLOps platforms.

Key Features:
- Unified API for all artifact types
- Intelligent routing to optimal backends
- Support for large files and external references
- Rich metadata and versioning

Usage:
    python basic_artifact_example.py
"""

import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tracelet import Experiment
from tracelet.core.artifacts import ArtifactType


def create_sample_files():
    """Create sample files for artifact demonstration."""
    temp_dir = Path(tempfile.mkdtemp(prefix="tracelet_artifacts_"))

    # Create a fake model file
    model_path = temp_dir / "trained_model.pth"
    with open(model_path, "w") as f:
        f.write("# Fake PyTorch model state dict\n{'layer1.weight': [1, 2, 3]}")

    # Create a config file
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write("learning_rate: 0.001\nbatch_size: 32\nepochs: 10\n")

    # Create a sample image
    image_path = temp_dir / "sample_prediction.png"
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(image_path)

    # Create a report
    report_path = temp_dir / "training_report.html"
    with open(report_path, "w") as f:
        f.write("""
        <html><body>
        <h1>Training Report</h1>
        <p>Final accuracy: 94.5%</p>
        <p>Training time: 2.5 hours</p>
        </body></html>
        """)

    return temp_dir, {"model": model_path, "config": config_path, "image": image_path, "report": report_path}


def demo_basic_artifacts():
    """Demonstrate basic artifact logging."""
    print("🎯 BASIC ARTIFACT LOGGING DEMO")
    print("=" * 50)

    # Create experiment with artifact tracking enabled
    exp = Experiment(
        name=f"artifact_demo_{int(time.time())}",
        backend=["mlflow"],  # Use MLflow only for reliable demo
        artifacts=True,  # Enable artifact tracking
        tags=["demo", "artifacts"],
    )

    try:
        exp.start()
        print("✅ Experiment started with artifact tracking")

        # Create sample files
        temp_dir, files = create_sample_files()
        print(f"📁 Created sample files in: {temp_dir}")

        # 1. Log a model artifact
        print("\n1. Logging model artifact...")
        model_artifact = exp.create_artifact(
            name="trained_classifier", artifact_type=ArtifactType.MODEL, description="Trained image classifier model"
        )

        # Add model file and metadata
        model_artifact.add_file(str(files["model"]), "model/classifier.pth")
        model_artifact.add_file(str(files["config"]), "model/config.yaml")

        # Add model metadata (skip actual model object for demo)
        model_artifact.metadata.update({
            "framework": "pytorch",
            "architecture": "CNN",
            "layers": ["conv1", "conv2", "fc1", "fc2"],
            "parameters": 125000000,
        })

        # Log to all backends
        results = exp.log_artifact(model_artifact)
        print(f"   ✅ Model logged to {len(results)} backends: {list(results.keys())}")

        # 2. Log an image artifact
        print("\n2. Logging image artifact...")
        image_artifact = exp.create_artifact(
            name="prediction_sample",
            artifact_type=ArtifactType.IMAGE,
            description="Sample model prediction visualization",
        ).add_file(str(files["image"]), "samples/prediction.png")

        # Add image metadata
        image_artifact.metadata.update({
            "prediction_class": "cat",
            "confidence": 0.945,
            "ground_truth": "cat",
            "correct": True,
        })

        results = exp.log_artifact(image_artifact)
        print(f"   ✅ Image logged to {len(results)} backends")

        # 3. Log a report artifact
        print("\n3. Logging report artifact...")
        report_artifact = exp.create_artifact(
            name="training_summary",
            artifact_type=ArtifactType.REPORT,
            description="Comprehensive training results and metrics",
        ).add_file(str(files["report"]), "reports/summary.html")

        # Add training summary as object
        training_summary = {
            "final_accuracy": 0.945,
            "final_loss": 0.123,
            "training_time_hours": 2.5,
            "epochs_completed": 10,
            "best_epoch": 8,
        }
        report_artifact.add_object(training_summary, "metrics", "json")

        results = exp.log_artifact(report_artifact)
        print(f"   ✅ Report logged to {len(results)} backends")

        # 4. Log external dataset reference
        print("\n4. Logging dataset reference...")
        dataset_artifact = exp.create_artifact(
            name="training_data", artifact_type=ArtifactType.DATASET, description="ImageNet training subset"
        )

        # Add external reference (use http URL to avoid S3 dependency)
        dataset_artifact.add_reference(
            "https://example.com/datasets/imagenet-subset.tar.gz",
            size_bytes=5 * 1024 * 1024 * 1024,  # 5GB
            description="Compressed training images",
        )

        # Add dataset metadata
        dataset_artifact.metadata.update({
            "num_samples": 50000,
            "num_classes": 1000,
            "image_size": "224x224",
            "format": "JPEG",
        })

        results = exp.log_artifact(dataset_artifact)
        print(f"   ✅ Dataset reference logged to {len(results)} backends")

        print("\n📊 ARTIFACTS LOGGED SUCCESSFULLY")
        print("=" * 50)
        print("✨ Check your backends to see the logged artifacts:")
        print("   • Models optimally routed to MLflow")
        print("   • Images optimally routed to W&B")
        print("   • Reports and configs distributed to all backends")
        print("   • Large dataset stored as reference, not uploaded")

        # Clean up temp files
        import shutil

        shutil.rmtree(temp_dir)
        print("\n🧹 Cleaned up temporary files")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        exp.stop()
        print("🔚 Experiment stopped")


def demo_artifact_retrieval():
    """Demonstrate artifact retrieval (when implemented)."""
    print("\n🔍 ARTIFACT RETRIEVAL DEMO")
    print("=" * 50)

    exp = Experiment(name="retrieval_demo", backend=["mlflow"], artifacts=True)

    try:
        exp.start()

        # Try to list artifacts (will show warning - not implemented yet)
        artifacts = exp.list_artifacts()
        print(f"📋 Found {len(artifacts)} artifacts")

        # Try to get specific artifact (will show warning - not implemented yet)
        artifact = exp.get_artifact("trained_classifier")
        if artifact:
            print(f"🎯 Retrieved artifact: {artifact.name}")
        else:
            print("⚠️  Artifact retrieval not yet implemented")

    finally:
        exp.stop()


if __name__ == "__main__":
    print("🚀 TRACELET ARTIFACT SYSTEM DEMO")
    print("This demo shows the new unified artifact tracking system\n")

    demo_basic_artifacts()
    demo_artifact_retrieval()

    print("\n🎉 Demo completed!")
    print("\nNext steps:")
    print("• Check your W&B/MLflow dashboards for logged artifacts")
    print("• Try the automagic artifact detection with Lightning")
    print("• Experiment with different artifact types and backends")
