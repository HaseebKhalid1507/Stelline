"""Tests for config module."""
from pathlib import Path
import pytest
import yaml


def test_default_config_has_required_fields():
    """Default config should have session_dir, sources, batch_size, model."""
    from stelline.config import StellineConfig
    
    config = StellineConfig()
    
    # Required fields must exist
    assert hasattr(config, 'session_dir')
    assert hasattr(config, 'sources')
    assert hasattr(config, 'batch_size')
    assert hasattr(config, 'model')
    
    # session_dir should be a Path
    assert isinstance(config.session_dir, Path)
    
    # sources should be a list
    assert isinstance(config.sources, list)
    
    # batch_size should be int > 0
    assert isinstance(config.batch_size, int)
    assert config.batch_size > 0
    
    # model should be non-empty string
    assert isinstance(config.model, str)
    assert config.model.strip() != ""


def test_env_vars_override_defaults():
    """Environment variables should override default config values."""
    import os
    from stelline.config import StellineConfig
    
    # Set environment variables
    original_session_dir = os.environ.get('STELLINE_SESSION_DIR')
    original_batch_size = os.environ.get('STELLINE_BATCH_SIZE')
    
    try:
        os.environ['STELLINE_SESSION_DIR'] = '/tmp/test_sessions'
        os.environ['STELLINE_BATCH_SIZE'] = '10'
        
        config = StellineConfig.from_env()
        
        assert str(config.session_dir) == '/tmp/test_sessions'
        assert config.batch_size == 10
        
    finally:
        # Cleanup
        if original_session_dir is None:
            os.environ.pop('STELLINE_SESSION_DIR', None)
        else:
            os.environ['STELLINE_SESSION_DIR'] = original_session_dir
            
        if original_batch_size is None:
            os.environ.pop('STELLINE_BATCH_SIZE', None)
        else:
            os.environ['STELLINE_BATCH_SIZE'] = original_batch_size


def test_yaml_config_loads_and_overrides_defaults(tmp_path):
    """YAML config should load and override default values."""
    from stelline.config import StellineConfig
    
    # Create a test YAML config
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
session_dir: /tmp/yaml_sessions
batch_size: 15
model: claude-3-sonnet
""")
    
    config = StellineConfig.load(str(config_file))
    
    assert str(config.session_dir) == "/tmp/yaml_sessions"
    assert config.batch_size == 15
    assert config.model == "claude-3-sonnet"


def test_invalid_config_handling(tmp_path):
    """Invalid config should raise appropriate errors."""
    from stelline.config import StellineConfig
    
    # Test missing file
    with pytest.raises(FileNotFoundError):
        StellineConfig.load("/nonexistent/config.yaml")
    
    # Test invalid YAML
    invalid_yaml_file = tmp_path / "invalid.yaml"
    invalid_yaml_file.write_text("invalid: yaml: content: [")
    
    with pytest.raises(yaml.YAMLError):
        StellineConfig.load(str(invalid_yaml_file))
    
    # Test invalid field types
    bad_config_file = tmp_path / "bad.yaml"
    bad_config_file.write_text("batch_size: not_a_number")
    
    with pytest.raises((TypeError, ValueError)):
        StellineConfig.load(str(bad_config_file))