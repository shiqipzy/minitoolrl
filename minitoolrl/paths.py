"""Local artifact locations, resolved relative to this package and project root."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
ARTIFACTS_ROOT = PACKAGE_ROOT / 'artifacts'
MODELS_ROOT = ARTIFACTS_ROOT / 'models'
RUNS_ROOT = ARTIFACTS_ROOT / 'runs'
CURRENT_RUN_ID = 'tool_composition_qwen3_registry_lr2e6'

