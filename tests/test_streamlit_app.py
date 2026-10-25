"""Smoke tests for the Streamlit dashboard and its sub-pages."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.config import load_config
from ui.pages.page_overview import render_page_overview
from ui.pages.page_energy_explorer import render_page_energy_explorer
from ui.pages.page_forecasting_studio import render_page_forecasting_studio
from ui.pages.page_battery_laboratory import render_page_battery_laboratory
from ui.pages.page_model_evaluation import render_page_model_evaluation
from ui.pages.page_methodology import render_page_methodology
from ui.pages.page_export import render_page_export


@pytest.fixture
def test_config():
    return load_config()


def test_pages_import_and_render_smoke(test_config):
    """Smoke test ensuring each page function can be invoked without unhandled exceptions."""
    pages = [
        render_page_overview,
        render_page_energy_explorer,
        render_page_forecasting_studio,
        render_page_battery_laboratory,
        render_page_model_evaluation,
        render_page_methodology,
        render_page_export,
    ]

    for page_fn in pages:
        # Should not raise exception
        try:
            page_fn(config=test_config, technical_mode=True)
            page_fn(config=test_config, technical_mode=False)
        except Exception as e:
            pytest.fail(f"Page function {page_fn.__name__} raised unexpected exception: {e}")
