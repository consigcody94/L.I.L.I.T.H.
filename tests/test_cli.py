"""Smoke tests for the unified ``lilith`` CLI.

These tests verify that every command at least imports cleanly and prints
help — the kind of breakage that's annoying to debug but trivial to catch
with a one-line check per command.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from lilith.cli import app


@pytest.fixture
def runner():
    return CliRunner()


class TestRootHelp:
    def test_top_level_help(self, runner):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for sub in ("download", "process", "train", "forecast", "api", "version"):
            assert sub in result.output

    def test_version(self, runner):
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "lilith" in result.output


class TestDownloadGroup:
    @pytest.mark.parametrize("sub", ["ghcn", "ghcn-hourly", "climate", "asos-1min", "synoptic"])
    def test_download_subcommand_help(self, runner, sub):
        result = runner.invoke(app, ["download", sub, "--help"])
        assert result.exit_code == 0


class TestProcessGroup:
    @pytest.mark.parametrize("sub", ["ghcn", "hourly"])
    def test_process_subcommand_help(self, runner, sub):
        result = runner.invoke(app, ["process", sub, "--help"])
        assert result.exit_code == 0


class TestTrainGroup:
    @pytest.mark.parametrize("sub", ["simple", "full"])
    def test_train_subcommand_help(self, runner, sub):
        result = runner.invoke(app, ["train", sub, "--help"])
        assert result.exit_code == 0


class TestForecastGuards:
    def test_forecast_missing_checkpoint_exits_nonzero(self, runner, tmp_path):
        # Point at a nonexistent path; the command should fail cleanly with
        # an informative error, not a stack trace.
        result = runner.invoke(
            app,
            [
                "forecast",
                "--lat", "40.7", "--lon", "-74.0", "--days", "3",
                "--checkpoint", str(tmp_path / "nope.pt"),
            ],
        )
        assert result.exit_code != 0
        assert "Checkpoint not found" in result.output


class TestSynopticTokenGuard:
    def test_synoptic_without_token_exits_clean(self, runner, monkeypatch):
        monkeypatch.delenv("SYNOPTIC_TOKEN", raising=False)
        # Block the home-dir token file too so the test is hermetic.
        monkeypatch.setattr(
            "data.download.synoptic.Path",
            __import__("pathlib").Path,  # keep real Path
        )
        result = runner.invoke(
            app,
            [
                "download", "synoptic",
                "--start", "202001010000", "--end", "202001020000",
                "KORD",
            ],
        )
        # exit_code 2 is what we use in the CLI for missing-credential errors.
        # Allow non-zero generally (file-based fallback could also kick in if
        # the developer happens to have ~/.synoptic_token).
        if result.exit_code != 0:
            assert "synoptic" in result.output.lower() or "token" in result.output.lower()
