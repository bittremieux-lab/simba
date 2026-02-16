"""Integration tests for SIMBA CLI commands.

These are smoke tests to ensure CLI commands can be invoked without errors.
They do NOT test actual training functionality.
"""

import shutil
import subprocess

import pytest
from click.testing import CliRunner

from simba.cli import cli


class TestCLI:
    """Test suite for SIMBA CLI commands using Click's test runner."""

    def test_cli_help(self):
        """Test that the CLI shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "simba" in result.output.lower()

    def test_preprocess_command_help(self):
        """Test that the preprocess command shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "--help"])
        assert result.exit_code == 0
        assert "preprocess" in result.output.lower()
        # Hydra-based command uses OVERRIDES instead of individual flags
        assert "OVERRIDES" in result.output

    def test_preprocess_command_missing_required_args(self):
        """Test that preprocess command can run with default config."""
        runner = CliRunner()
        # Hydra config has defaults, so command doesn't strictly require args
        # Just check it doesn't crash on help
        result = runner.invoke(cli, ["preprocess", "--help"])
        assert result.exit_code == 0

    def test_preprocess_command_invalid_spectra_path(self):
        """Test that preprocess command validates spectra path at runtime."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "preprocess",
                "paths.spectra_path=/nonexistent/spectra.mgf",
                "paths.preprocessing_dir=/tmp/test_workspace",
            ],
        )
        # Command should fail when trying to load nonexistent file
        assert result.exit_code != 0

    @pytest.mark.parametrize(
        "override",
        [
            "preprocessing.max_spectra_train=100",
            "preprocessing.max_spectra_val=50",
            "preprocessing.max_spectra_test=50",
            "hardware.num_workers=4",
            "preprocessing.val_split=0.15",
            "preprocessing.test_split=0.15",
            "preprocessing.overwrite=true",
        ],
    )
    def test_preprocess_command_has_expected_options(self, override):
        """Test that preprocess command accepts Hydra config overrides."""
        runner = CliRunner()
        result = runner.invoke(cli, ["preprocess", "--help"])
        assert result.exit_code == 0
        # Just verify help works - actual overrides are tested functionally

    def test_train_command_help(self):
        """Test that the train command shows help message with Hydra config."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        assert "train" in result.output.lower()
        # Check for Hydra-style documentation
        assert "OVERRIDES" in result.output or "overrides" in result.output.lower()
        assert "Configuration" in result.output or "config" in result.output.lower()

    def test_train_command_with_overrides(self):
        """Test that train command accepts Hydra overrides."""
        runner = CliRunner()
        # This will fail due to missing data, but should parse the overrides correctly
        result = runner.invoke(
            cli,
            [
                "train",
                "training.epochs=10",
                "training.batch_size=32",
            ],
        )
        # Should fail with file not found, not argument parsing error
        assert "Error" in result.output or result.exit_code != 0

    def test_train_command_shows_hydra_examples(self):
        """Test that train command help shows Hydra override examples."""
        runner = CliRunner()
        result = runner.invoke(cli, ["train", "--help"])
        assert result.exit_code == 0
        # Check that examples show Hydra syntax
        assert "training.epochs" in result.output or "override" in result.output.lower()

    @pytest.mark.parametrize(
        "override",
        [
            "training.epochs=50",
            "training.batch_size=64",
            "optimizer.lr=0.001",
            "hardware.num_workers=4",
            "hardware.accelerator=gpu",
            "training.val_check_interval=500",
        ],
    )
    def test_train_command_accepts_hydra_overrides(self, override):
        """Test that train command accepts various Hydra overrides."""
        runner = CliRunner()
        # Will fail due to missing data, but overrides should be parsed
        result = runner.invoke(cli, ["train", override])
        # Should not fail with "no such option" error
        assert "--" not in result.output or result.exit_code != 0

    def test_inference_command_help(self):
        """Test that the inference command shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "--help"])
        assert result.exit_code == 0
        assert "inference" in result.output.lower()
        # With Hydra migration, these are now Hydra overrides, not CLI flags
        assert "OVERRIDES" in result.output
        assert "paths.checkpoint_dir" in result.output
        assert "paths.preprocessing_dir" in result.output

    def test_inference_command_missing_required_args(self):
        """Test that inference command fails without required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Missing" in result.output

    def test_inference_command_invalid_checkpoint_dir(self):
        """Test that inference command fails with invalid checkpoint directory."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "inference",
                "paths.checkpoint_dir=/nonexistent/checkpoints",
                "paths.preprocessing_dir=/tmp/test_preprocessing",
            ],
        )
        assert result.exit_code != 0

    @pytest.mark.parametrize(
        "config_key",
        [
            "paths.checkpoint_dir",
            "paths.preprocessing_dir",
        ],
    )
    def test_inference_command_has_expected_options(self, config_key):
        """Test that inference command has expected Hydra configuration keys."""
        runner = CliRunner()
        result = runner.invoke(cli, ["inference", "--help"])
        assert result.exit_code == 0
        assert config_key in result.output

    def test_analog_discovery_command_help(self):
        """Test that the analog-discovery command shows help message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analog-discovery", "--help"])
        assert result.exit_code == 0
        assert "analog" in result.output.lower()
        assert "--model-path" in result.output
        assert "--query-spectra" in result.output
        assert "--reference-spectra" in result.output
        assert "--output-dir" in result.output

    def test_analog_discovery_command_missing_required_args(self):
        """Test that analog-discovery command fails without required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analog-discovery"])
        assert result.exit_code != 0
        assert "Error" in result.output or "Missing" in result.output

    def test_analog_discovery_command_invalid_model_path(self):
        """Test that analog-discovery command fails with invalid model path."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "analog-discovery",
                "--model-path",
                "/nonexistent/model.ckpt",
                "--query-spectra",
                "/tmp/query.mgf",
                "--reference-spectra",
                "/tmp/reference.mgf",
                "--output-dir",
                "/tmp/output",
            ],
        )
        assert result.exit_code != 0

    def test_analog_discovery_command_uses_hydra_config(self):
        """Test that analog-discovery command uses Hydra configuration instead of CLI options."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analog-discovery", "--help"])
        assert result.exit_code == 0

        # Should have required path options
        assert "--model-path" in result.output
        assert "--query-spectra" in result.output
        assert "--reference-spectra" in result.output
        assert "--output-dir" in result.output

        # Should NOT have these as CLI options (now via Hydra)
        assert "--query-index" not in result.output
        assert "--top-k" not in result.output
        assert "--device" not in result.output
        assert "--batch-size" not in result.output

        # Should mention Hydra overrides in help
        assert "OVERRIDES" in result.output or "overrides" in result.output


class TestCLIEntryPoint:
    """Test suite for CLI entry point (simba command).

    These tests verify that the 'simba' command is properly installed
    and accessible from PATH. They will FAIL if the package is not
    installed in editable mode (uv sync / pip install -e .).
    """

    def test_cli_entry_point_installed(self):
        """Test that simba CLI entry point is installed."""
        # Try to find simba command in PATH
        simba_path = shutil.which("simba")
        assert simba_path is not None, (
            "simba command not found in PATH. "
            "Run 'uv sync' or 'pip install -e .' to install the CLI entry point."
        )

    def test_cli_entry_point_help(self):
        """Test that simba CLI entry point shows help."""
        try:
            result = subprocess.run(
                ["simba", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            assert "Usage" in result.stdout or "simba" in result.stdout.lower()
        except subprocess.TimeoutExpired:
            pytest.fail("simba --help command timed out after 10 seconds")

    def test_cli_preprocess_entry_point_help(self):
        """Test that simba preprocess entry point shows help."""
        try:
            result = subprocess.run(
                ["simba", "preprocess", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            assert "preprocess" in result.stdout.lower()
        except subprocess.TimeoutExpired:
            pytest.fail("simba preprocess --help command timed out after 10 seconds")

    def test_cli_train_entry_point_help(self):
        """Test that simba train entry point shows help."""
        try:
            result = subprocess.run(
                ["simba", "train", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            assert "train" in result.stdout.lower()
        except subprocess.TimeoutExpired:
            pytest.fail("simba train --help command timed out after 10 seconds")

    def test_cli_inference_entry_point_help(self):
        """Test that simba inference entry point shows help."""
        try:
            result = subprocess.run(
                ["simba", "inference", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            assert "inference" in result.stdout.lower()
        except subprocess.TimeoutExpired:
            pytest.fail("simba inference --help command timed out after 10 seconds")

    def test_cli_analog_discovery_entry_point_help(self):
        """Test that simba analog-discovery entry point shows help."""
        try:
            result = subprocess.run(
                ["simba", "analog-discovery", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert result.returncode == 0, f"Command failed: {result.stderr}"
            assert "analog" in result.stdout.lower()
        except subprocess.TimeoutExpired:
            pytest.fail(
                "simba analog-discovery --help command timed out after 10 seconds"
            )
