from __future__ import annotations

import functools
import importlib
from typing import TYPE_CHECKING, Any

from ..core.interfaces import FrameworkInterface
from ..core.orchestrator import MetricType

if TYPE_CHECKING:
    from ..core.experiment import Experiment


class PyTorchFramework(FrameworkInterface):
    """PyTorch framework integration that patches tensorboard for metric tracking"""

    def __init__(self, patch_tensorboard: bool = True):
        self._experiment = None
        self._original_add_scalar = None
        self._original_add_scalars = None
        self._original_add_histogram = None
        self._original_add_image = None
        self._original_add_text = None
        self._original_add_figure = None
        self._original_add_embedding = None
        self._original_add_video = None
        self._original_add_audio = None
        self._original_add_mesh = None
        self._original_add_hparams = None
        self._patch_tensorboard = patch_tensorboard
        self._tensorboard_available = self._check_tensorboard()

    @staticmethod
    def _check_tensorboard():
        """Check if tensorboard is available"""
        try:
            importlib.import_module("torch.utils.tensorboard")
        except ImportError:
            return False
        else:
            return True

    def initialize(self, experiment: Experiment):
        self._experiment = experiment
        if self._patch_tensorboard and self._tensorboard_available:
            self._patch_tensorboard_writer()

    def start_tracking(self):
        pass  # Nothing specific needed for PyTorch

    def stop_tracking(self):
        if self._patch_tensorboard and self._tensorboard_available:
            self._unpatch_tensorboard_writer()

    def log_metric(self, name: str, value: Any, iteration: int):
        if self._experiment:
            self._experiment.log_metric(name, value, iteration)

    def log_enhanced_metric(
        self, name: str, value: Any, metric_type: MetricType, iteration: int, metadata: dict | None = None
    ):
        """Log an enhanced metric with specific type and metadata"""
        if self._experiment:
            from ..core.orchestrator import MetricData

            metric = MetricData(
                name=name,
                value=value,
                type=metric_type,
                iteration=iteration,
                source=self._experiment.get_source_id(),
                metadata=metadata or {},
            )
            self._experiment.emit_metric(metric)

    def _patch_tensorboard_writer(self):
        """Patch tensorboard's SummaryWriter to capture all metric types"""
        if not self._original_add_scalar:
            summary_writer = self._get_summary_writer()
            if not summary_writer:
                return

            self._store_original_methods(summary_writer)
            self._apply_scalar_patches(summary_writer)
            self._apply_enhanced_patches(summary_writer)

    def _get_summary_writer(self):
        """Get SummaryWriter class or None if not available"""
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self._tensorboard_available = False
            return None
        else:
            return SummaryWriter

    def _store_original_methods(self, summary_writer):
        """Store references to original methods"""
        self._original_add_scalar = summary_writer.add_scalar
        self._original_add_scalars = summary_writer.add_scalars

        # Store enhanced methods with fallback for optional methods
        self._original_add_histogram = getattr(summary_writer, "add_histogram", None)
        self._original_add_image = getattr(summary_writer, "add_image", None)
        self._original_add_text = getattr(summary_writer, "add_text", None)
        self._original_add_figure = getattr(summary_writer, "add_figure", None)
        self._original_add_embedding = getattr(summary_writer, "add_embedding", None)
        self._original_add_video = getattr(summary_writer, "add_video", None)
        self._original_add_audio = getattr(summary_writer, "add_audio", None)
        self._original_add_mesh = getattr(summary_writer, "add_mesh", None)
        self._original_add_hparams = getattr(summary_writer, "add_hparams", None)

    def _apply_scalar_patches(self, summary_writer):
        """Apply patches for scalar methods"""

        @functools.wraps(summary_writer.add_scalar)
        def wrapped_add_scalar(
            writer_self, tag: str, scalar_value: float, global_step: int | None = None, *args, **kwargs
        ):
            result = self._original_add_scalar(writer_self, tag, scalar_value, global_step, *args, **kwargs)
            self.log_metric(tag, scalar_value, global_step)
            return result

        @functools.wraps(summary_writer.add_scalars)
        def wrapped_add_scalars(
            writer_self,
            main_tag: str,
            tag_scalar_dict: dict[str, float],
            global_step: int | None = None,
            *args,
            **kwargs,
        ):
            result = self._original_add_scalars(writer_self, main_tag, tag_scalar_dict, global_step, *args, **kwargs)
            for tag, scalar in tag_scalar_dict.items():
                metric_name = f"{main_tag}/{tag}"
                self.log_metric(metric_name, scalar, global_step)
            return result

        summary_writer.add_scalar = wrapped_add_scalar
        summary_writer.add_scalars = wrapped_add_scalars

    def _apply_enhanced_patches(self, summary_writer):
        """Apply patches for enhanced visualization methods"""
        enhanced_methods = [
            ("add_histogram", MetricType.HISTOGRAM, self._create_histogram_wrapper),
            ("add_image", MetricType.IMAGE, self._create_image_wrapper),
            ("add_text", MetricType.TEXT, self._create_text_wrapper),
            ("add_figure", MetricType.FIGURE, self._create_figure_wrapper),
            ("add_embedding", MetricType.EMBEDDING, self._create_embedding_wrapper),
            ("add_video", MetricType.VIDEO, self._create_video_wrapper),
            ("add_audio", MetricType.AUDIO, self._create_audio_wrapper),
            ("add_mesh", MetricType.MESH, self._create_mesh_wrapper),
            ("add_hparams", MetricType.HPARAMS, self._create_hparams_wrapper),
        ]

        for method_name, metric_type, wrapper_creator in enhanced_methods:
            original = getattr(self, f"_original_{method_name}", None)
            if original:
                wrapper = wrapper_creator(original, metric_type)
                setattr(summary_writer, method_name, wrapper)

    def _create_histogram_wrapper(self, original, metric_type):
        """Create wrapper for histogram logging"""

        @functools.wraps(original)
        def wrapped(writer_self, tag, values, global_step=None, bins="tensorflow", *args, **kwargs):
            result = original(writer_self, tag, values, global_step, bins, *args, **kwargs)
            self.log_enhanced_metric(
                tag, values, metric_type, global_step, {"bins": bins, "shape": getattr(values, "shape", None)}
            )
            return result

        return wrapped

    def _create_image_wrapper(self, original, metric_type):
        """Create wrapper for image logging"""

        @functools.wraps(original)
        def wrapped(writer_self, tag, img_tensor, global_step=None, walltime=None, dataformats="CHW", *args, **kwargs):
            result = original(writer_self, tag, img_tensor, global_step, walltime, dataformats, *args, **kwargs)
            self.log_enhanced_metric(
                tag,
                img_tensor,
                metric_type,
                global_step,
                {"dataformats": dataformats, "shape": getattr(img_tensor, "shape", None)},
            )
            return result

        return wrapped

    def _create_text_wrapper(self, original, metric_type):
        """Create wrapper for text logging"""

        @functools.wraps(original)
        def wrapped(writer_self, tag, text_string, global_step=None, walltime=None, *args, **kwargs):
            result = original(writer_self, tag, text_string, global_step, walltime, *args, **kwargs)
            self.log_enhanced_metric(tag, text_string, metric_type, global_step)
            return result

        return wrapped

    def _create_figure_wrapper(self, original, metric_type):
        """Create wrapper for figure logging"""

        @functools.wraps(original)
        def wrapped(writer_self, tag, figure, global_step=None, close=True, walltime=None, *args, **kwargs):
            result = original(writer_self, tag, figure, global_step, close, walltime, *args, **kwargs)
            self.log_enhanced_metric(tag, figure, metric_type, global_step, {"close": close})
            return result

        return wrapped

    def _create_embedding_wrapper(self, original, metric_type):
        """Create wrapper for embedding logging"""

        @functools.wraps(original)
        def wrapped(writer_self, mat, metadata=None, label_img=None, global_step=None, tag="default", *args, **kwargs):
            result = original(writer_self, mat, metadata, label_img, global_step, tag, *args, **kwargs)
            self.log_enhanced_metric(
                tag,
                mat,
                metric_type,
                global_step,
                {"has_metadata": metadata is not None, "has_label_img": label_img is not None},
            )
            return result

        return wrapped

    def _create_video_wrapper(self, original, metric_type):
        """Create wrapper for video logging"""

        @functools.wraps(original)
        def wrapped(writer_self, tag, vid_tensor, global_step=None, fps=4, walltime=None, *args, **kwargs):
            result = original(writer_self, tag, vid_tensor, global_step, fps, walltime, *args, **kwargs)
            self.log_enhanced_metric(
                tag, vid_tensor, metric_type, global_step, {"fps": fps, "shape": getattr(vid_tensor, "shape", None)}
            )
            return result

        return wrapped

    def _create_audio_wrapper(self, original, metric_type):
        """Create wrapper for audio logging"""

        @functools.wraps(original)
        def wrapped(writer_self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None, *args, **kwargs):
            result = original(writer_self, tag, snd_tensor, global_step, sample_rate, walltime, *args, **kwargs)
            self.log_enhanced_metric(
                tag,
                snd_tensor,
                metric_type,
                global_step,
                {"sample_rate": sample_rate, "shape": getattr(snd_tensor, "shape", None)},
            )
            return result

        return wrapped

    def _create_mesh_wrapper(self, original, metric_type):
        """Create wrapper for mesh logging"""

        @functools.wraps(original)
        def wrapped(
            writer_self,
            tag,
            vertices,
            colors=None,
            faces=None,
            config_dict=None,
            global_step=None,
            walltime=None,
            *args,
            **kwargs,
        ):
            result = original(
                writer_self, tag, vertices, colors, faces, config_dict, global_step, walltime, *args, **kwargs
            )
            self.log_enhanced_metric(
                tag,
                vertices,
                metric_type,
                global_step,
                {"has_colors": colors is not None, "has_faces": faces is not None, "config": config_dict},
            )
            return result

        return wrapped

    def _create_hparams_wrapper(self, original, metric_type):
        """Create wrapper for hyperparameters logging"""

        @functools.wraps(original)
        def wrapped(writer_self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None, *args, **kwargs):
            result = original(writer_self, hparam_dict, metric_dict, hparam_domain_discrete, run_name, *args, **kwargs)
            self.log_enhanced_metric(
                "hparams", {"hparams": hparam_dict, "metrics": metric_dict}, metric_type, None, {"run_name": run_name}
            )
            return result

        return wrapped

    def _unpatch_tensorboard_writer(self):
        """Restore original tensorboard methods"""
        if self._original_add_scalar and self._tensorboard_available:
            summary_writer = self._get_summary_writer()
            if summary_writer:
                self._restore_original_methods(summary_writer)
            self._clear_method_references()

    def _restore_original_methods(self, summary_writer):
        """Restore original TensorBoard methods"""
        # Restore basic methods
        summary_writer.add_scalar = self._original_add_scalar
        summary_writer.add_scalars = self._original_add_scalars

        # Restore enhanced methods if they were patched
        enhanced_methods = [
            "add_histogram",
            "add_image",
            "add_text",
            "add_figure",
            "add_embedding",
            "add_video",
            "add_audio",
            "add_mesh",
            "add_hparams",
        ]

        for method_name in enhanced_methods:
            original = getattr(self, f"_original_{method_name}", None)
            if original:
                setattr(summary_writer, method_name, original)

    def _clear_method_references(self):
        """Clear all stored method references"""
        method_names = [
            "_original_add_scalar",
            "_original_add_scalars",
            "_original_add_histogram",
            "_original_add_image",
            "_original_add_text",
            "_original_add_figure",
            "_original_add_embedding",
            "_original_add_video",
            "_original_add_audio",
            "_original_add_mesh",
            "_original_add_hparams",
        ]

        for method_name in method_names:
            setattr(self, method_name, None)
