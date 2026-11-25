"""Wake the Streamlit Community Cloud app with a real browser session."""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

APP_URL = os.environ.get("STREAMLIT_APP_URL", "").strip()
WAKE_BUTTON_NAMES = [
    "Yes, get this app back up!",
    "Yes, get this app back up",
    "Get this app back up",
    "Wake up",
]
NAVIGATION_ATTEMPTS = 3


def click_wake_button(page) -> bool:
    for name in WAKE_BUTTON_NAMES:
        button = page.get_by_role("button", name=name)

        try:
            button.wait_for(state="visible", timeout=5_000)
            button.click()
            print(f"Clicked wake button: {name}")
            return True
        except PlaywrightTimeoutError:
            continue
    return False


def wait_until_ready(page) -> None:
    # Streamlit renders these containers when the app frontend is available.
    page.locator('[data-testid="stAppViewContainer"]').wait_for(
        state="visible",
        timeout=180_000,
    )
    page.locator('[data-testid="stMain"]').wait_for(
        state="visible",
        timeout=180_000,
    )


def main() -> int:
    if not APP_URL:
        print(
            "STREAMLIT_APP_URL is not set. Add it as a repository Actions "
            "variable before running this workflow."
        )
        return 2

    parsed_url = urlparse(APP_URL)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        print("STREAMLIT_APP_URL must be a valid http:// or https:// URL.")
        return 2

    screenshot_path = Path("wake-streamlit.png")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_default_timeout(30_000)

        print(f"Opening {APP_URL}")
        try:
            for attempt in range(1, NAVIGATION_ATTEMPTS + 1):
                try:
                    page.goto(
                        APP_URL,
                        wait_until="domcontentloaded",
                        timeout=60_000,
                    )
                    break
                except PlaywrightTimeoutError:
                    if attempt == NAVIGATION_ATTEMPTS:
                        page.screenshot(path=str(screenshot_path), full_page=True)
                        print("Failed to load the Streamlit page.")
                        print(f"Page title: {page.title()}")
                        print(f"Saved screenshot to {screenshot_path}")
                        return 1

                    print(
                        "Navigation timed out; retrying "
                        f"({attempt}/{NAVIGATION_ATTEMPTS - 1})..."
                    )

            woke = click_wake_button(page)
            if not woke:
                print(
                    "No wake button found; "
                    "checking whether the app is already live."
                )
            else:
                print("Wake button clicked. Waiting for the app to start...")

            try:
                wait_until_ready(page)
            except PlaywrightTimeoutError:
                page.screenshot(path=str(screenshot_path), full_page=True)
                print("Timed out waiting for the Streamlit app to become ready.")
                print(f"Page title: {page.title()}")
                print(f"Saved screenshot to {screenshot_path}")
                return 1

            print("Streamlit app is live.")
            return 0
        finally:
            browser.close()


if __name__ == "__main__":
    sys.exit(main())
