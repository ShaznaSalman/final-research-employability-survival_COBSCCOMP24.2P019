"""Colour tokens in frontend/ui.py: text stays readable (WCAG AA) on every surface, in light and dark themes."""
from __future__ import annotations

import re

import pytest

from frontend import ui


def _tokens(css: str) -> dict[str, str]:
    return dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-fA-F]{6})", css))


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4   # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def _ratio(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


LIGHT = {**_tokens(ui.TOKENS_LIGHT), "white": "#ffffff"}
THEMES = {"light": LIGHT, "dark": {**LIGHT, **_tokens(ui.TOKENS_DARK)}}
SURFACES = ["wash", "panel", "blue-soft", "mint-soft", "peach-soft", "lav-soft"]
TEXT_PAIRS = ([("ink", s) for s in SURFACES] + [("muted", s) for s in SURFACES]
              + [("brand-ink", "blue-soft"), ("brand-ink", "wash"), ("mint-ink", "mint-soft"),
                 ("peach-ink", "peach-soft"), ("lav-ink", "lav-soft"), ("amber-ink", "amber-soft"),
                 ("on-brand", "brand"), ("on-accent", "accent"),
                 ("hero-ink", "hero-a"), ("hero-ink", "hero-b"), ("hero-muted", "hero-a"), ("hero-muted", "hero-b"),
                 ("hero-kicker", "hero-a"), ("hero-kicker", "hero-b"),
                 ("white", "b-blue"), ("white", "b-mint"), ("white", "b-peach"), ("white", "b-lav")])


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fg,bg", TEXT_PAIRS)
def test_text_contrast_is_at_least_aa(theme, fg, bg):
    tokens = THEMES[theme]
    assert _ratio(tokens[fg], tokens[bg]) >= 4.5, f"{fg} on {bg} in {theme}: {_ratio(tokens[fg], tokens[bg]):.2f}"


@pytest.mark.parametrize("theme", THEMES)
def test_brand_colour_is_visible_against_page(theme):
    tokens = THEMES[theme]
    assert _ratio(tokens["brand"], tokens["wash"]) >= 3.0      # focus rings, filled controls
