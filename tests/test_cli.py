"""Tests for CLI module.""" 
import pytest
import tempfile
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock


def test_cli_shows_help():
    """CLI should show help text."""
    from stelline.cli import cli
    
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    
    assert result.exit_code == 0
    assert "Stelline" in result.output
    assert "Session Intelligence Tool" in result.output


def test_scan_command_shows_unprocessed_sessions():
    """Scan command should show unprocessed session counts by source."""
    from stelline.cli import cli
    
    mock_stats = {
        "jawz": {"total": 10, "processed": 7, "unprocessed": 3},
        "dexter": {"total": 5, "processed": 5, "unprocessed": 0}
    }
    
    with patch("stelline.cli.SessionDiscovery") as mock_discovery_class:
        mock_discovery = Mock()
        mock_discovery.get_source_stats.return_value = mock_stats
        mock_discovery_class.return_value = mock_discovery
        
        with patch("stelline.cli.SessionTracker") as mock_tracker_class:
            runner = CliRunner()
            result = runner.invoke(cli, ['scan'])
            
            assert result.exit_code == 0
            assert "jawz" in result.output
            assert "3 unprocessed" in result.output
            assert "dexter" in result.output
            assert "0 unprocessed" in result.output
            assert "Total unprocessed: 3" in result.output


def test_status_command_shows_system_status():
    """Status command should show overall system status and stats."""
    from stelline.cli import cli
    
    mock_stats = {
        "overall": {
            "total_processed": 100,
            "total_memories": 250,
            "failed_count": 2,
            "avg_duration": 3.5
        },
        "by_source": [
            {"source": "jawz", "processed": 80, "memories": 200},
            {"source": "dexter", "processed": 20, "memories": 50}
        ]
    }
    
    with patch("stelline.cli.SessionTracker") as mock_tracker_class:
        mock_tracker = Mock()
        mock_tracker.get_stats.return_value = mock_stats
        mock_tracker_class.return_value = mock_tracker
        
        runner = CliRunner()
        result = runner.invoke(cli, ['status'])
        
        assert result.exit_code == 0
        assert "100" in result.output  # total_processed
        assert "250" in result.output  # total_memories
        assert "2" in result.output    # failed_count
        assert "3.5s" in result.output # avg_duration
        assert "jawz" in result.output
        assert "dexter" in result.output


def test_harvest_command_processes_sessions():
    """Harvest command should process unprocessed sessions."""
    from stelline.cli import cli
    
    # Mock session file
    mock_session = Mock()
    mock_session.session_id = "test_123"
    mock_session.path = Path("/test.jsonl")
    
    # Mock dependencies
    with patch("stelline.cli.SessionDiscovery") as mock_discovery_class, \
         patch("stelline.cli.StellinePipeline") as mock_pipeline_class, \
         patch("stelline.cli.SessionTracker") as mock_tracker_class, \
         patch("stelline.cli.ContextLoader") as mock_context_class:
        
        mock_discovery = Mock()
        mock_discovery.discover_unprocessed.return_value = [mock_session]
        mock_discovery_class.return_value = mock_discovery
        
        mock_pipeline = Mock()
        mock_pipeline.process_session.return_value = {
            "status": "success",
            "memories_extracted": 3
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_tracker = Mock()
        mock_tracker.start_harvest_run.return_value = 1
        mock_tracker_class.return_value = mock_tracker
        
        runner = CliRunner()
        result = runner.invoke(cli, ['harvest'])
        
        assert result.exit_code == 0
        assert "Processing 1 sessions" in result.output
        assert "Processed 1 sessions" in result.output
        
        # Should have processed the session
        mock_pipeline.process_session.assert_called_once_with(mock_session, dry_run=False)


def test_harvest_command_dry_run():
    """Harvest command should support dry run mode."""
    from stelline.cli import cli
    
    mock_session = Mock()
    mock_session.session_id = "test_123"
    
    with patch("stelline.cli.SessionDiscovery") as mock_discovery_class, \
         patch("stelline.cli.StellinePipeline") as mock_pipeline_class, \
         patch("stelline.cli.SessionTracker") as mock_tracker_class, \
         patch("stelline.cli.ContextLoader") as mock_context_class:
        
        mock_discovery = Mock()
        mock_discovery.discover_unprocessed.return_value = [mock_session]
        mock_discovery_class.return_value = mock_discovery
        
        mock_pipeline = Mock()
        mock_pipeline.process_session.return_value = {
            "status": "dry_run",
            "transcript_chars": 1000
        }
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_tracker = Mock()
        mock_tracker.start_harvest_run.return_value = 1
        mock_tracker_class.return_value = mock_tracker
        
        runner = CliRunner()
        result = runner.invoke(cli, ['harvest', '--dry-run'])
        
        assert result.exit_code == 0
        assert "Would process" in result.output
        
        # Should have called with dry_run=True
        mock_pipeline.process_session.assert_called_once_with(mock_session, dry_run=True)


def test_harvest_command_batch_limit():
    """Harvest command should respect batch size limit."""
    from stelline.cli import cli
    
    # Create multiple mock sessions
    mock_sessions = [Mock(session_id=f"test_{i}") for i in range(10)]
    
    with patch("stelline.cli.SessionDiscovery") as mock_discovery_class, \
         patch("stelline.cli.StellinePipeline") as mock_pipeline_class, \
         patch("stelline.cli.SessionTracker") as mock_tracker_class, \
         patch("stelline.cli.ContextLoader") as mock_context_class:
        
        mock_discovery = Mock()
        mock_discovery.discover_unprocessed.return_value = mock_sessions
        mock_discovery_class.return_value = mock_discovery
        
        mock_pipeline = Mock()
        mock_pipeline.process_session.return_value = {"status": "success", "memories_extracted": 1}
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_tracker = Mock()
        mock_tracker.start_harvest_run.return_value = 1
        mock_tracker_class.return_value = mock_tracker
        
        runner = CliRunner()
        result = runner.invoke(cli, ['harvest', '--batch', '3'])
        
        assert result.exit_code == 0
        assert "Processing 3 sessions" in result.output
        
        # Should have processed only 3 sessions
        assert mock_pipeline.process_session.call_count == 3


def test_harvest_command_source_filter():
    """Harvest command should filter by source when specified."""
    from stelline.cli import cli
    
    mock_session = Mock()
    mock_session.session_id = "test_123"
    
    with patch("stelline.cli.SessionDiscovery") as mock_discovery_class, \
         patch("stelline.cli.StellinePipeline") as mock_pipeline_class, \
         patch("stelline.cli.SessionTracker") as mock_tracker_class, \
         patch("stelline.cli.ContextLoader") as mock_context_class:
        
        mock_discovery = Mock()
        mock_discovery.discover_unprocessed.return_value = [mock_session]
        mock_discovery_class.return_value = mock_discovery
        
        mock_pipeline = Mock()
        mock_pipeline.process_session.return_value = {"status": "success", "memories_extracted": 1}
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        runner = CliRunner()
        result = runner.invoke(cli, ['harvest', '--source', 'dexter'])
        
        assert result.exit_code == 0
        
        # Should have called discover with source filter
        mock_discovery.discover_unprocessed.assert_called_once_with("dexter")


def test_harvest_command_specific_file():
    """Harvest command should process specific file when provided."""
    from stelline.cli import cli
    from stelline.discovery import SessionFile
    
    with tempfile.NamedTemporaryFile(suffix='.jsonl') as tmpfile:
        with patch("stelline.cli.StellinePipeline") as mock_pipeline_class, \
             patch("stelline.cli.SessionTracker") as mock_tracker_class, \
             patch("stelline.cli.ContextLoader") as mock_context_class, \
             patch("stelline.discovery.SessionFile.from_path") as mock_from_path:
            
            mock_session = Mock()
            mock_session.session_id = "file_123"
            mock_from_path.return_value = mock_session
            
            mock_pipeline = Mock()
            mock_pipeline.process_session.return_value = {"status": "success"}
            mock_pipeline_class.return_value = mock_pipeline
            
            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker
            
            runner = CliRunner()
            result = runner.invoke(cli, ['harvest', '--file', tmpfile.name])
            
            assert result.exit_code == 0
            assert f"Processed {mock_session.session_id}" in result.output
            
            # Should have processed the specific file
            mock_pipeline.process_session.assert_called_once_with(mock_session, dry_run=False)


def test_history_command_shows_recent_runs():
    """History command should show recent harvest run history."""
    from stelline.cli import cli
    
    mock_runs = [
        {
            "started_at": "2024-12-23T12:00:00",
            "status": "completed", 
            "sessions_processed": 5,
            "total_memories": 15
        },
        {
            "started_at": "2024-12-23T10:00:00",
            "status": "completed",
            "sessions_processed": 3,
            "total_memories": 8
        }
    ]
    
    with patch("stelline.cli.SessionTracker") as mock_tracker_class:
        mock_tracker = Mock()
        mock_tracker.get_recent_runs.return_value = mock_runs
        mock_tracker_class.return_value = mock_tracker
        
        runner = CliRunner()
        result = runner.invoke(cli, ['history'])
        
        assert result.exit_code == 0
        assert "Recent Harvest Runs" in result.output
        assert "2024-12-23T12:00:00" in result.output
        assert "completed" in result.output
        assert "5" in result.output
        assert "15" in result.output


def test_sources_command_shows_configured_sources():
    """Sources command should show configured sources and settings."""
    from stelline.cli import cli
    from stelline.config import StellineConfig, SourceConfig
    
    # Mock config with custom sources
    mock_config = StellineConfig()
    mock_config.sources = [
        SourceConfig("jawz", "--home-haseeb--", enabled=True),
        SourceConfig("dexter", "--home-haseeb-Dexter--", enabled=False, 
                    memkoshi_storage="~/.memkoshi-trading")
    ]
    mock_config.memkoshi_storage = "~/.memkoshi"
    mock_config.batch_size = 5
    mock_config.model = "claude-3-haiku-20240307"
    
    with patch("stelline.cli.StellineConfig") as mock_config_class:
        mock_config_class.return_value = mock_config
        
        runner = CliRunner()
        result = runner.invoke(cli, ['sources'], obj={'config': mock_config})
        
        assert result.exit_code == 0
        assert "Configured Sources" in result.output
        assert "jawz" in result.output
        assert "enabled" in result.output
        assert "dexter" in result.output
        assert "disabled" in result.output
        assert "Custom storage: ~/.memkoshi-trading" in result.output
        assert "Default storage: ~/.memkoshi" in result.output
        assert "Batch size: 5" in result.output


def test_cli_handles_no_unprocessed_sessions():
    """Harvest command should handle case with no unprocessed sessions."""
    from stelline.cli import cli
    
    with patch("stelline.cli.SessionDiscovery") as mock_discovery_class, \
         patch("stelline.cli.SessionTracker") as mock_tracker_class, \
         patch("stelline.cli.ContextLoader") as mock_context_class:
        
        mock_discovery = Mock()
        mock_discovery.discover_unprocessed.return_value = []  # No sessions
        mock_discovery_class.return_value = mock_discovery
        
        mock_tracker = Mock()
        mock_tracker_class.return_value = mock_tracker
        
        runner = CliRunner()
        result = runner.invoke(cli, ['harvest'])
        
        assert result.exit_code == 0
        assert "No unprocessed sessions found" in result.output


def test_cli_config_loading():
    """CLI should load config properly."""
    from stelline.cli import cli
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmpfile:
        tmpfile.write("""
session_dir: /tmp/test_sessions
batch_size: 10
""")
        config_path = tmpfile.name
    
    try:
        with patch("stelline.cli.SessionTracker") as mock_tracker_class:
            runner = CliRunner()
            result = runner.invoke(cli, ['--config', config_path, 'scan'])
            
            assert result.exit_code == 0
    finally:
        Path(config_path).unlink()


def test_harvest_tracks_run_metadata():
    """Harvest command should track run metadata."""
    from stelline.cli import cli
    
    mock_session = Mock()
    mock_session.session_id = "test_123"
    
    with patch("stelline.cli.SessionDiscovery") as mock_discovery_class, \
         patch("stelline.cli.StellinePipeline") as mock_pipeline_class, \
         patch("stelline.cli.SessionTracker") as mock_tracker_class, \
         patch("stelline.cli.ContextLoader") as mock_context_class:
        
        mock_discovery = Mock()
        mock_discovery.discover_unprocessed.return_value = [mock_session]
        mock_discovery_class.return_value = mock_discovery
        
        mock_pipeline = Mock()
        mock_pipeline.process_session.return_value = {"status": "success", "memories_extracted": 2}
        mock_pipeline_class.return_value = mock_pipeline
        
        mock_tracker = Mock()
        mock_tracker.start_harvest_run.return_value = 1
        mock_tracker_class.return_value = mock_tracker
        
        runner = CliRunner()
        result = runner.invoke(cli, ['harvest', '--dry-run'])
        
        # Should start tracking with metadata
        mock_tracker.start_harvest_run.assert_called_once()
        call_args = mock_tracker.start_harvest_run.call_args[0]
        metadata = call_args[0]
        assert metadata["batch_size"] == 1
        assert metadata["dry_run"] == True
        
        # Should complete tracking
        mock_tracker.complete_harvest_run.assert_called_once()