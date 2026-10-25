"""Automated Playwright script to capture screenshots of all 7 Streamlit pages across Light and Dark themes at 1366x768 and 1920x1080."""
import os
import time
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOT_DIR = ROOT / "assets" / "screenshots"

PAGES = [
    ("Overview", "01_overview.png"),
    ("Energy Explorer", "02_energy_explorer.png"),
    ("Forecasting Studio", "03_forecasting_studio.png"),
    ("Battery Laboratory", "04_battery_laboratory.png"),
    ("Model Evaluation", "05_model_evaluation.png"),
    ("Methodology & Explainability", "06_methodology.png"),
    ("Data & Results Export", "07_export.png"),
]

THEMES = [
    ("light", "Light Mode (Default)"),
    ("dark", "Dark Mode"),
]

RESOLUTIONS = [
    (1366, 768),
    (1920, 1080),
]


def capture_all_screenshots():
    url = "http://localhost:8502"
    print(f"Connecting to Streamlit at {url}...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for theme_key, theme_label in THEMES:
            for width, height in RESOLUTIONS:
                res_dir = f"{width}x{height}"
                out_dir = SCREENSHOT_DIR / theme_key / res_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n--- Capturing Theme: {theme_key} | Resolution: {res_dir} ---")

                context = browser.new_context(viewport={"width": width, "height": height})
                page = context.new_page()
                page.goto(url, wait_until="networkidle")
                time.sleep(2)

                # Select Theme
                try:
                    theme_radio = page.locator("label").filter(has_text=theme_label)
                    if theme_radio.count() > 0:
                        theme_radio.first.click()
                        time.sleep(1.5)
                except Exception as e:
                    print(f"Warning setting theme: {e}")

                # Capture each page
                for page_name, filename in PAGES:
                    print(f"  Capturing {page_name}...")
                    try:
                        nav_radio = page.locator("label").filter(has_text=page_name)
                        if nav_radio.count() > 0:
                            nav_radio.first.click()
                            time.sleep(2.5)  # Wait for charts to animate

                        out_path = out_dir / filename
                        page.screenshot(path=str(out_path), full_page=False)
                        print(f"    Saved: {out_path.relative_to(ROOT)}")
                    except Exception as e:
                        print(f"    Error capturing {page_name}: {e}")

                context.close()

        browser.close()
    print("\nAll screenshots captured successfully!")


if __name__ == "__main__":
    capture_all_screenshots()
