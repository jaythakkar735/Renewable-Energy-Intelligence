"""Captures real screenshots of Streamlit dashboard pages using headless Chrome."""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

screenshots_dir = Path("reports/screenshots")
screenshots_dir.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path="/usr/bin/google-chrome",
        headless=True,
        args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
    )
    context = browser.new_context(viewport={"width": 1366, "height": 768})
    page = context.new_page()

    print("Navigating to http://localhost:8502...")
    page.goto("http://localhost:8502", wait_until="networkidle", timeout=30000)
    time.sleep(3)  # Allow WebSocket and Plotly to settle

    # 1. Page: Overview
    print("Capturing 01_overview.png...")
    page.screenshot(path=str(screenshots_dir / "01_overview.png"))

    # 2. Page: Energy Explorer
    print("Navigating to Energy Explorer...")
    page.locator('label:has-text("Energy Explorer")').click()
    time.sleep(3)
    page.screenshot(path=str(screenshots_dir / "02_energy_explorer.png"))

    # 3. Page: Forecasting Studio
    print("Navigating to Forecasting Studio...")
    page.locator('label:has-text("Forecasting Studio")').click()
    time.sleep(3)
    page.screenshot(path=str(screenshots_dir / "03_forecasting_studio.png"))

    # 4. Page: Battery Laboratory
    print("Navigating to Battery Laboratory...")
    page.locator('label:has-text("Battery Laboratory")').click()
    time.sleep(4)  # Allow live LP simulation to execute and Plotly charts to render
    page.screenshot(path=str(screenshots_dir / "04_battery_laboratory.png"))

    # 5. Page: Model Evaluation
    print("Navigating to Model Evaluation...")
    page.locator('label:has-text("Model Evaluation")').click()
    time.sleep(3)
    page.screenshot(path=str(screenshots_dir / "05_model_evaluation.png"))

    # 6. Page: Methodology
    print("Navigating to Methodology...")
    page.locator('label:has-text("Methodology & Explainability")').click()
    time.sleep(3)
    page.screenshot(path=str(screenshots_dir / "06_methodology.png"))

    # 7. Page: Export
    print("Navigating to Export...")
    page.locator('label:has-text("Data & Results Export")').click()
    time.sleep(2)
    page.screenshot(path=str(screenshots_dir / "07_export.png"))

    browser.close()
    print("Successfully captured all 7 screenshots in reports/screenshots/!")
