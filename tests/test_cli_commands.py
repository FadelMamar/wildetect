import os
import shutil
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner
from wildetect.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "wildetect version" in result.output


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "WildDetect - Wildlife Detection System" in result.output


def test_clear_results_cancel():
    result = runner.invoke(
        app, ["clear-results", "--results-dir", "some_dir"], input="n\n"
    )
    assert result.exit_code == 0
    assert "Operation cancelled" in result.output


def test_clear_results_success(tmp_path):
    # Create a dummy results directory
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "dummy.txt").write_text("test")
    result = runner.invoke(
        app, ["clear-results", "--results-dir", str(results_dir)], input="y\n"
    )
    assert result.exit_code == 0
    assert "have been deleted" in result.output
    assert not results_dir.exists()


def test_clear_results_nonexistent():
    result = runner.invoke(
        app, ["clear-results", "--results-dir", "nonexistent_dir"], input="y\n"
    )
    assert result.exit_code == 0
    assert "does not exist" in result.output
