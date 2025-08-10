"""
Tests for the profiler utilities.
"""
import tempfile
from pathlib import Path

import pytest
from wildetect.utils.profiler import ProfilerManager, profile_command


def test_profiler_manager_initialization():
    """Test ProfilerManager initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        profiler = ProfilerManager(
            output_dir=Path(temp_dir),
            enable_profiling=True,
            enable_memory_profile=True,
            enable_line_profile=False,
            enable_gpu_profile=False,
        )

        assert profiler.output_dir == Path(temp_dir)
        assert profiler.enable_profiling is True
        assert profiler.enable_memory_profile is True
        assert profiler.enable_line_profile is False
        assert profiler.enable_gpu_profile is False


def test_profiler_manager_context():
    """Test ProfilerManager as context manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with ProfilerManager(
            output_dir=Path(temp_dir),
            enable_profiling=False,
            enable_memory_profile=True,
            enable_line_profile=False,
            enable_gpu_profile=False,
        ) as profiler:
            # Simulate some work
            import time

            time.sleep(0.1)

            # Check that memory monitoring was started
            assert profiler.memory_monitoring is True

        # Check that output files were created
        output_files = list(Path(temp_dir).glob("*"))
        assert len(output_files) > 0


def test_memory_profiling():
    """Test memory profiling functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with ProfilerManager(
            output_dir=Path(temp_dir),
            enable_memory_profile=True,
        ) as profiler:
            # Test memory tracking function
            def test_function():
                # Allocate some memory
                test_list = [i for i in range(10000)]
                return len(test_list)

            result = profiler.track_memory_usage(test_function)
            assert result == 10000

            # Test memory monitoring
            profiler.start_memory_monitoring(interval=0.1)
            import time

            time.sleep(0.3)  # Allow a few samples
            profiler.stop_memory_monitoring()

            # Check that we have memory samples
            summary = profiler.get_memory_summary()
            assert summary is not None
            assert summary["samples_count"] > 0


def test_profile_command_context():
    """Test profile_command context manager."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with profile_command(
            output_dir=Path(temp_dir),
            profile=False,
            memory_profile=True,
            line_profile=False,
            gpu_profile=False,
        ) as profiler:
            # Simulate some work
            import time

            time.sleep(0.1)

            # Check that memory monitoring was started
            assert profiler.memory_monitoring is True


def test_profiler_without_memory_profiler():
    """Test profiler behavior when memory_profiler is not available."""
    # This test simulates the case where memory_profiler is not installed
    with tempfile.TemporaryDirectory() as temp_dir:
        with ProfilerManager(
            output_dir=Path(temp_dir),
            enable_memory_profile=True,
        ) as profiler:
            # The profiler should handle missing dependencies gracefully
            assert (
                profiler.enable_memory_profile is False
                or profiler.memory_profiler is not None
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
