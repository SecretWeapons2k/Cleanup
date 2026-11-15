"""Regression tests for external Yabai input paths."""
from yolo_mask_auto_select_modules import config


def test_yabai_input_paths_match_current_repository_layout():
    assert config.DEFAULT_CSV_DIR == config.YABAI_PROJECT_ROOT / "output" / "csv"
    assert config.DEFAULT_ANCHORS_CSV == config.DEFAULT_CSV_DIR / "anchors.csv"
    assert config.DEFAULT_IMAGES_DIR == config.YABAI_PROJECT_ROOT / "mit_kanji"
