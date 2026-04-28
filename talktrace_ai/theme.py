from .paths import resource_path


def load_obsidian_css() -> str:
    return resource_path("static/obsidian_theme.css").read_text(encoding="utf-8")
