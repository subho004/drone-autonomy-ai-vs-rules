"""Capture the screenshots and demo video used in the README (assets/dashboard and assets/demo).

Drives a real Chromium (Playwright) against a running dashboard and the static reports.
Start the server first: ``.venv/bin/uvicorn main:app --port 8765``. Needs
``requirements/docs.txt`` and ``playwright install chromium``; ffmpeg turns the recording
into MP4 and GIF. Constants at the top instead of CLI args.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from playwright.sync_api import Page, Playwright, ViewportSize, sync_playwright

from app.core.logging import configure_logging, get_logger

BASE_URL = "http://127.0.0.1:8765"
MEDIA_DIR = ROOT_DIR / "assets"
SHOT_DIR = MEDIA_DIR / "dashboard"
DEMO_DIR = MEDIA_DIR / "demo"
REPORT_DIR = ROOT_DIR / "reports"
DESKTOP: ViewportSize = {"width": 1440, "height": 900}
VIDEO_SIZE: ViewportSize = {"width": 1280, "height": 760}
MOBILE: ViewportSize = {"width": 390, "height": 844}
LIVE_EXPERIMENT = "live"
PILOT_EXPERIMENT = "pilot"
# A scenario where Jev (601 ms) dodges an approaching obstacle and GPT-6 Luna (1080 ms) does not.
REPLAY_PAIR = (17497, 17857)
SETTLE_S = 1.2
REPORT_VIEW: ViewportSize = {"width": 1100, "height": 900}
GIF_WIDTH = 760
GIF_FPS = 8

logger = get_logger("capture_docs_media")


def experiment_id(page: Page, name: str) -> str:
    """Value of the experiment <option> whose label starts with ``name``."""

    return page.evaluate(
        """(name) => { const o = [...document.querySelectorAll('#expSelect option')]
            .find(o => o.textContent.trim().toLowerCase().startsWith(name)); return o ? o.value : null; }""",
        name,
    )


def pick_experiment(page: Page, name: str) -> None:
    value = experiment_id(page, name)
    if value is None:
        raise SystemExit(f"experiment {name!r} not found in the dashboard")
    page.select_option("#expSelect", value=value)
    page.wait_for_timeout(int(SETTLE_S * 1000))


def goto(page: Page, hash_: str = "") -> None:
    page.goto(f"{BASE_URL}/dashboard{hash_}", wait_until="networkidle")
    page.wait_for_timeout(int(SETTLE_S * 1000))


def open_replay(page: Page) -> None:
    """Open the compare replay under the live experiment (hash-only navigation keeps state)."""

    goto(page)
    pick_experiment(page, LIVE_EXPERIMENT)
    page.evaluate(f"location.hash = '#replay/{REPLAY_PAIR[0]},{REPLAY_PAIR[1]}'")
    page.wait_for_timeout(int(SETTLE_S * 1000 * 1.5))


def shot(page: Page, name: str, *, full: bool = False) -> None:
    path = SHOT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=full)
    logger.info("wrote %s", path.relative_to(ROOT_DIR))


def scroll_to(page: Page, selector: str) -> None:
    page.locator(selector).first.scroll_into_view_if_needed()
    page.wait_for_timeout(600)


def scrub_to(page: Page, fraction: float) -> None:
    page.evaluate(
        """(f) => { const s = document.querySelector('#scrub'); s.value = (parseFloat(s.max) * f).toFixed(3);
            s.dispatchEvent(new Event('input', {bubbles: true})); }""",
        fraction,
    )
    page.wait_for_timeout(500)


def take_screenshots(pw: Playwright) -> None:
    browser = pw.chromium.launch()
    light = browser.new_context(
        viewport=DESKTOP, color_scheme="light", device_scale_factor=1.5
    )
    page = light.new_page()
    goto(page)
    pick_experiment(page, LIVE_EXPERIMENT)
    shot(page, "dashboard-live-overview")
    page.mouse.wheel(0, 640)
    page.wait_for_timeout(700)
    shot(page, "dashboard-live-tiers")
    pick_experiment(page, PILOT_EXPERIMENT)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(700)
    shot(page, "dashboard-latency-sweep")
    page.click("a[data-tab=flights]")
    page.wait_for_timeout(1000)
    shot(page, "dashboard-flights")
    open_replay(page)
    scrub_to(page, 0.30)
    page.mouse.wheel(0, 260)
    page.wait_for_timeout(500)
    shot(page, "replay-compare-before")
    scrub_to(page, 0.62)
    shot(page, "replay-compare-collision")
    scroll_to(page, ".panes")
    shot(page, "replay-decision-gantt", full=True)
    page.close()
    light.close()
    dark = browser.new_context(
        viewport=DESKTOP, color_scheme="dark", device_scale_factor=1.5
    )
    page = dark.new_page()
    goto(page)
    pick_experiment(page, LIVE_EXPERIMENT)
    shot(page, "dashboard-dark")
    dark.close()
    phone = browser.new_context(
        viewport=MOBILE, color_scheme="light", device_scale_factor=2, is_mobile=True
    )
    page = phone.new_page()
    goto(page)
    pick_experiment(page, LIVE_EXPERIMENT)
    shot(page, "dashboard-mobile")
    phone.close()
    report = browser.new_context(viewport=REPORT_VIEW, device_scale_factor=1.5)
    page = report.new_page()
    page.goto((REPORT_DIR / "live.html").as_uri())
    page.wait_for_timeout(800)
    shot(page, "static-report")
    report.close()
    browser.close()


def record_video(pw: Playwright) -> Path:
    tmp = DEMO_DIR / "_video"
    shutil.rmtree(tmp, ignore_errors=True)
    browser = pw.chromium.launch()
    ctx = browser.new_context(
        viewport=VIDEO_SIZE,
        color_scheme="light",
        record_video_dir=str(tmp),
        record_video_size=VIDEO_SIZE,
    )
    page = ctx.new_page()
    goto(page)
    pick_experiment(page, LIVE_EXPERIMENT)
    page.wait_for_timeout(2500)
    for dy in (420, 420, 420):  # slow scroll through the charts
        page.mouse.wheel(0, dy)
        page.wait_for_timeout(1500)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(600)
    pick_experiment(page, PILOT_EXPERIMENT)
    page.wait_for_timeout(2500)
    page.click("a[data-tab=flights]")
    page.wait_for_timeout(2000)
    open_replay(page)
    page.mouse.wheel(0, 200)
    page.wait_for_timeout(1200)
    page.click("#btnPlay")
    page.wait_for_timeout(9500)  # play the whole (about 8 s) flight in real time
    scrub_to(page, 0.62)
    page.wait_for_timeout(2500)
    video = page.video
    ctx.close()
    browser.close()
    assert video is not None
    return Path(video.path())


def encode(webm: Path) -> None:
    mp4, gif = DEMO_DIR / "dashboard-tour.mp4", DEMO_DIR / "dashboard-tour.gif"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(webm),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    palette = DEMO_DIR / "_palette.png"
    scale = f"fps={GIF_FPS},scale={GIF_WIDTH}:-1:flags=lanczos"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(webm),
            "-vf",
            f"{scale},palettegen=stats_mode=diff",
            str(palette),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(webm),
            "-i",
            str(palette),
            "-lavfi",
            f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
            str(gif),
        ],
        check=True,
        capture_output=True,
    )
    palette.unlink(missing_ok=True)
    for f in (mp4, gif):
        logger.info(
            "wrote %s (%.1f MB)", f.relative_to(ROOT_DIR), f.stat().st_size / 1e6
        )


def main() -> None:
    configure_logging()
    start = time.perf_counter()
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("media capture started against %s", BASE_URL)
    with sync_playwright() as pw:
        take_screenshots(pw)
        webm = record_video(pw)
    encode(webm)
    shutil.rmtree(DEMO_DIR / "_video", ignore_errors=True)
    logger.info("media capture finished in %.1fs", time.perf_counter() - start)


if __name__ == "__main__":
    main()
