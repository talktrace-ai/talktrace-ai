from .paths import resource_path


def load_theme_css() -> str:
    return resource_path("static/theme.css").read_text(encoding="utf-8")
