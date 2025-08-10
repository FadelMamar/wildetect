"""
Profiling utilities for CLI commands.
"""
import cProfile
import pstats
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from .rich_logging import (
    console,
    create_table,
    print_error,
    print_info,
    print_success,
    print_warning,
)


class ProfilerManager:
    """Manages different types of profiling for CLI commands."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        enable_profiling: bool = False,
        enable_memory_profile: bool = False,
        enable_line_profile: bool = False,
        enable_gpu_profile: bool = False,
    ):
        self.output_dir = output_dir or Path(".")
        self.enable_profiling = enable_profiling
        self.enable_memory_profile = enable_memory_profile
        self.enable_line_profile = enable_line_profile
        self.enable_gpu_profile = enable_gpu_profile

        # Profiler instances
        self.profiler = None
        self.line_profiler = None
        self.memory_profiler = None
        self.start_time = None
        self.memory_samples = []
        self.memory_monitoring = False

        # Rich progress tracking
        self.progress = None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        """Start profiling when entering context."""
        self.start_time = time.time()

        # Display startup banner
        from wildetect.utils.rich_logging import Panel

        console.print(
            Panel.fit(
                "[bold blue]🚀 Profiling Session Started[/bold blue]",
                border_style="blue",
            )
        )

        if self.enable_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            print_success("Standard profiling enabled")

        if self.enable_memory_profile:
            try:
                from memory_profiler import profile

                # Create a memory profiler that will track memory usage
                self.memory_profiler = profile
                print_success("Memory profiling enabled")
                # Start memory monitoring
                self.start_memory_monitoring()
            except ImportError:
                print_warning(
                    "memory_profiler not installed. Install with: pip install memory_profiler"
                )
                self.enable_memory_profile = False

        if self.enable_line_profile:
            try:
                from line_profiler import LineProfiler

                self.line_profiler = LineProfiler()
                print_success("Line profiling enabled - this will be slower")
            except ImportError:
                print_warning(
                    "line_profiler not installed. Install with: pip install line_profiler"
                )
                self.enable_line_profile = False

        if self.enable_gpu_profile:
            try:
                import torch

                if torch.cuda.is_available():
                    print_success("GPU profiling enabled")
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                else:
                    print_warning("CUDA not available, GPU profiling disabled")
                    self.enable_gpu_profile = False
            except ImportError:
                print_warning("PyTorch not available, GPU profiling disabled")
                self.enable_gpu_profile = False

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop profiling and save results when exiting context."""
        end_time = time.time()
        execution_time = end_time - self.start_time if self.start_time else 0

        # Display completion summary
        print_success(f"✨ Command completed in {execution_time:.2f} seconds")

        # Stop memory monitoring if it was running
        if self.enable_memory_profile and self.memory_monitoring:
            self.stop_memory_monitoring()

        # Handle standard profiling
        if self.enable_profiling and self.profiler:
            self._save_standard_profile()

        # Handle memory profiling
        if self.enable_memory_profile and self.memory_profiler:
            self._save_memory_profile()

        # Handle line profiling
        if self.enable_line_profile and self.line_profiler:
            self._save_line_profile()

        # Handle GPU profiling
        if self.enable_gpu_profile:
            self._print_gpu_stats()

        # Display final summary
        self._display_final_summary(execution_time)

    def _display_final_summary(self, execution_time: float):
        """Display a beautiful final summary of the profiling session."""
        summary_data = {
            "Execution Time": f"{execution_time:.2f} seconds",
            "Standard Profiling": "✅ Enabled"
            if self.enable_profiling
            else "❌ Disabled",
            "Memory Profiling": "✅ Enabled"
            if self.enable_memory_profile
            else "❌ Disabled",
            "Line Profiling": "✅ Enabled"
            if self.enable_line_profile
            else "❌ Disabled",
            "GPU Profiling": "✅ Enabled" if self.enable_gpu_profile else "❌ Disabled",
            "Output Directory": str(self.output_dir),
        }

        from wildetect.utils.rich_logging import create_summary_panel

        panel = create_summary_panel("📊 Profiling Session Summary", summary_data)
        console.print(panel)

    def create_progress_tracker(self, description: str = "Processing"):
        """Create a Rich progress tracker for long-running operations."""
        from wildetect.utils.rich_logging import ProgressTracker

        self.progress = ProgressTracker(description)
        return self.progress

    def profile_function(self, func: Callable, *args, **kwargs):
        """Profile a specific function if line profiling is enabled."""
        if self.enable_line_profile and self.line_profiler:
            self.line_profiler.add_function(func)
            self.line_profiler.enable_by_count()

        result = func(*args, **kwargs)

        if self.enable_line_profile and self.line_profiler:
            self.line_profiler.disable_by_count()

        return result

    def memory_profile_function(self, func: Callable, *args, **kwargs):
        """Profile memory usage of a specific function if memory profiling is enabled."""
        if self.enable_memory_profile and self.memory_profiler:
            # Apply the memory profiler decorator to the function
            profiled_func = self.memory_profiler(func)
            return profiled_func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    def track_memory_usage(self, func: Callable, *args, **kwargs):
        """Track memory usage before and after function execution."""
        if not self.enable_memory_profile:
            return func(*args, **kwargs)

        try:
            import psutil

            process = psutil.Process()

            # Get memory before function execution
            memory_before = process.memory_info()

            # Execute function
            result = func(*args, **kwargs)

            # Get memory after function execution
            memory_after = process.memory_info()

            # Calculate memory difference
            memory_diff = memory_after.rss - memory_before.rss

            # Log memory usage with Rich formatting
            print_info(f"💾 Memory before: {memory_before.rss / 1024**2:.2f} MB")
            print_info(f"💾 Memory after: {memory_after.rss / 1024**2:.2f} MB")
            print_info(f"💾 Memory difference: {memory_diff / 1024**2:.2f} MB")

            return result

        except ImportError:
            print_warning("psutil not available for memory tracking")
            return func(*args, **kwargs)

    def start_memory_monitoring(self, interval: float = 1.0):
        """Start periodic memory monitoring."""
        if not self.enable_memory_profile:
            return

        try:
            import threading
            import time

            import psutil

            self.memory_monitoring = True
            self.memory_samples = []

            def monitor_memory():
                process = psutil.Process()
                while self.memory_monitoring:
                    try:
                        memory_info = process.memory_info()
                        sample = {
                            "timestamp": time.time(),
                            "rss": memory_info.rss,
                            "vms": memory_info.vms,
                            "percent": process.memory_percent(),
                        }
                        self.memory_samples.append(sample)
                        time.sleep(interval)
                    except Exception as e:
                        print_warning(f"Error during memory monitoring: {e}")
                        break

            # Start monitoring in a separate thread
            self.monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
            self.monitor_thread.start()
            print_success(f"🔍 Memory monitoring started with {interval}s interval")

        except ImportError:
            print_warning("psutil not available for memory monitoring")

    def stop_memory_monitoring(self):
        """Stop periodic memory monitoring."""
        self.memory_monitoring = False
        if hasattr(self, "monitor_thread"):
            self.monitor_thread.join(timeout=1.0)
        print_error("🛑 Memory monitoring stopped")

    def get_memory_summary(self):
        """Get a summary of memory usage from samples."""
        if not self.memory_samples:
            return None

        try:
            rss_values = [sample["rss"] for sample in self.memory_samples]
            vms_values = [sample["vms"] for sample in self.memory_samples]
            percent_values = [sample["percent"] for sample in self.memory_samples]

            summary = {
                "samples_count": len(self.memory_samples),
                "rss_min": min(rss_values) / 1024**2,  # MB
                "rss_max": max(rss_values) / 1024**2,  # MB
                "rss_avg": sum(rss_values) / len(rss_values) / 1024**2,  # MB
                "vms_min": min(vms_values) / 1024**2,  # MB
                "vms_max": max(vms_values) / len(vms_values) / 1024**2,  # MB
                "vms_avg": sum(vms_values) / len(vms_values) / 1024**2,  # MB
                "percent_min": min(percent_values),
                "percent_max": max(percent_values),
                "percent_avg": sum(percent_values) / len(percent_values),
            }

            return summary

        except Exception as e:
            print_warning(f"Error calculating memory summary: {e}")
            return None

    def _save_standard_profile(self):
        """Save standard profiling results."""
        self.profiler.disable()
        stats = pstats.Stats(self.profiler)

        # Save profiling results
        profile_path = self.output_dir / "profile_results.prof"
        stats.dump_stats(str(profile_path))

        # Create Rich table for profiling results
        from wildetect.utils.rich_logging import print_section_header

        print_section_header("📈 PROFILING RESULTS")

        # Get top 20 functions by cumulative time
        stats.sort_stats("cumulative")

        # Create table for top functions
        columns = [
            {"header": "Rank", "style": "cyan", "no_wrap": True},
            {"header": "Function", "style": "green"},
            {"header": "Calls", "style": "yellow", "justify": "right"},
            {"header": "Cumulative Time", "style": "magenta", "justify": "right"},
            {"header": "Per Call", "style": "blue", "justify": "right"},
        ]

        # Prepare data for table
        table_data = []
        # Get the top functions from stats
        top_functions = sorted(
            [(key, stats.stats[key]) for key in stats.stats.keys()],
            key=lambda x: x[1][3],  # Sort by cumulative time (index 3)
            reverse=True,
        )[:20]

        for i, ((filename, lineno, name), stats_data) in enumerate(top_functions):
            calls = stats_data[0]
            cumulative = stats_data[3]
            per_call = cumulative / calls if calls > 0 else 0

            table_data.append(
                [
                    str(i + 1),
                    f"{name}",
                    str(calls),
                    f"{cumulative:.4f}s",
                    f"{per_call:.6f}s",
                ]
            )

        table = create_table("Top Functions by Cumulative Time", columns, table_data)
        console.print(table)
        print_info(f"💾 Detailed profile saved to: {profile_path}")
        print_info(
            f"🔍 To view with snakeviz: pip install snakeviz && snakeviz {profile_path}"
        )

    def _save_memory_profile(self):
        """Save memory profiling results."""
        try:
            import psutil
            from memory_profiler import show_results

            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()

            print_section_header("💾 MEMORY PROFILING RESULTS")

            # Create memory summary table
            columns = [
                {"header": "Metric", "style": "cyan", "no_wrap": True},
                {"header": "Value", "style": "magenta"},
            ]

            # Prepare data for table
            table_data = [
                ["Current Memory", f"{memory_info.rss / 1024**3:.2f} GB"],
                ["Peak Memory", f"{memory_info.peak_wset / 1024**3:.2f} GB"],
            ]

            # Get memory summary if available
            memory_summary = self.get_memory_summary()
            if memory_summary:
                table_data.extend(
                    [
                        ["Samples Count", str(memory_summary["samples_count"])],
                        [
                            "RSS Range",
                            f"{memory_summary['rss_min']:.2f} - {memory_summary['rss_max']:.2f} MB",
                        ],
                        ["RSS Average", f"{memory_summary['rss_avg']:.2f} MB"],
                        [
                            "VMS Range",
                            f"{memory_summary['vms_min']:.2f} - {memory_summary['vms_max']:.2f} MB",
                        ],
                        ["VMS Average", f"{memory_summary['vms_avg']:.2f} MB"],
                        [
                            "CPU % Range",
                            f"{memory_summary['percent_min']:.1f}% - {memory_summary['percent_max']:.1f}%",
                        ],
                        ["CPU % Average", f"{memory_summary['percent_avg']:.1f}%"],
                    ]
                )

            table = create_table("Memory Usage Summary", columns, table_data)
            console.print(table)

            # Save memory profile results
            memory_profile_path = self.output_dir / "memory_profile_results.txt"
            with open(memory_profile_path, "w") as f:
                f.write("=== MEMORY PROFILING RESULTS ===\n")
                f.write(f"Current Memory Usage: {memory_info.rss / 1024**3:.2f} GB\n")
                f.write(
                    f"Peak Memory Usage: {memory_info.peak_wset / 1024**3:.2f} GB\n"
                )
                f.write(f"Memory Info: {memory_info}\n")

                if memory_summary:
                    f.write(
                        f"\nMemory Summary (from {memory_summary['samples_count']} samples):\n"
                    )
                    f.write(
                        f"  RSS: {memory_summary['rss_min']:.2f} - {memory_summary['rss_max']:.2f} MB (avg: {memory_summary['rss_avg']:.2f} MB)\n"
                    )
                    f.write(
                        f"  VMS: {memory_summary['vms_min']:.2f} - {memory_summary['vms_max']:.2f} MB (avg: {memory_summary['vms_avg']:.2f} MB)\n"
                    )
                    f.write(
                        f"  Percent: {memory_summary['percent_min']:.1f}% - {memory_summary['percent_max']:.1f}% (avg: {memory_summary['percent_avg']:.1f}%)\n"
                    )

                # Save memory samples if available
                if self.memory_samples:
                    f.write(f"\nDetailed Memory Samples:\n")
                    for i, sample in enumerate(self.memory_samples):
                        f.write(
                            f"  Sample {i+1}: RSS={sample['rss']/1024**2:.2f}MB, VMS={sample['vms']/1024**2:.2f}MB, Percent={sample['percent']:.1f}%\n"
                        )

            print_info(f"💾 Memory profile saved to: {memory_profile_path}")

        except ImportError:
            print_warning(
                "memory_profiler or psutil not available for detailed memory analysis"
            )

    def _save_line_profile(self):
        """Save line profiling results."""
        self.line_profiler.print_stats()

        line_profile_path = self.output_dir / "line_profile_results.txt"
        with open(line_profile_path, "w") as f:
            self.line_profiler.print_stats(stream=f)
        print_info(f"💾 Line profile saved to: {line_profile_path}")

    def _print_gpu_stats(self):
        """Print GPU memory statistics."""
        try:
            import torch

            if torch.cuda.is_available():
                columns = [
                    {"header": "Metric", "style": "cyan", "no_wrap": True},
                    {"header": "Value", "style": "magenta"},
                ]

                table_data = [
                    [
                        "Peak Memory",
                        f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
                    ],
                    [
                        "Current Memory",
                        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
                    ],
                ]

                table = create_table("🖥️ GPU Memory Statistics", columns, table_data)
                console.print(table)
        except ImportError:
            pass


class NullProfilerManager:
    """Null object for when no profiling is desired."""

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def profile_function(self, func: Callable, *args, **kwargs):
        """No-op function profiling."""
        return func(*args, **kwargs)

    def memory_profile_function(self, func: Callable, *args, **kwargs):
        """No-op memory profiling."""
        return func(*args, **kwargs)

    def track_memory_usage(self, func: Callable, *args, **kwargs):
        """No-op memory tracking."""
        return func(*args, **kwargs)

    def start_memory_monitoring(self, interval: float = 1.0):
        """No-op memory monitoring."""
        pass

    def stop_memory_monitoring(self):
        """No-op memory monitoring stop."""
        pass

    def get_memory_summary(self):
        """No-op memory summary."""
        return None


@contextmanager
def profile_command(
    output_dir: Optional[Path] = None,
    profile: bool = False,
    memory_profile: bool = False,
    line_profile: bool = False,
    gpu_profile: bool = False,
):
    """Context manager for profiling CLI commands.

    Returns a null context manager if no profiling is enabled,
    otherwise returns a full ProfilerManager.
    """
    # If no profiling is enabled, return a null context manager
    if not any([profile, memory_profile, line_profile, gpu_profile]):
        yield NullProfilerManager()
        return

    # Otherwise, create a full profiler manager
    profiler_manager = ProfilerManager(
        output_dir=output_dir,
        enable_profiling=profile,
        enable_memory_profile=memory_profile,
        enable_line_profile=line_profile,
        enable_gpu_profile=gpu_profile,
    )

    with profiler_manager as pm:
        yield pm
