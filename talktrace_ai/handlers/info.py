"""Info tab: Entwickler, Lizenz, Banner."""
from ._common import *


def register(state):
    output = state.output
    t = state.t

    @render.text
    def loc_title_info():
        return t("info", "tab_title")

    @render.ui
    def loc_info_dev_heading():
        return ui.p(t("info", "dev_heading"))

    @render.ui
    def loc_info_dev_body():
        def _link(href, label):
            return ui.tags.a(label, href=href, target="_blank", rel="noopener noreferrer")

        return ui.div(
            ui.tags.h6(t("info", "dev_current"), " (TalkTrace AI neo)"),
            ui.tags.p(
                _link("https://github.com/MoominVibeCoder", "Simon Filler"),
                " · ",
                _link("https://orcid.org/0009-0008-8736-8831", "ORCID"),
                " · ",
                _link("mailto:simon.filler@tu-dortmund.de", "simon.filler@tu-dortmund.de"),
            ),
            ui.tags.h6(
                t("info", "dev_origin"),
                " (",
                _link("https://github.com/talktrace-ai/talktrace-ai", "TalkTrace AI"),
                ")",
                style="margin-top:1rem;",
            ),
            ui.tags.p(
                _link(
                    "https://www.sozphil.uni-leipzig.de/institut-fuer-politikwissenschaft/arbeitsbereiche/professur-fuer-fachdidaktik-gemeinschaftskunde/team/prof-dr-dennis-hauk",
                    "Dennis Hauk",
                ),
                " · ",
                _link("https://orcid.org/0000-0002-5779-2876", "ORCID"),
                ui.tags.br(),
                _link("https://github.com/xrtze", "Jami Schorling"),
                " · ",
                _link("https://orcid.org/0009-0005-9007-2896", "ORCID"),
            ),
        )

    @render.ui
    def loc_info_license_heading():
        return ui.p(t("info", "license_heading"))

    @render.ui
    def loc_info_license_body():
        return ui.div(
            ui.tags.a(
                ui.tags.img(
                    src="/tt-assets/cc-by-nc.png",
                    alt="CC BY-NC 4.0",
                    style="border:0;display:block;margin-bottom:0.75rem;",
                ),
                href="https://creativecommons.org/licenses/by-nc/4.0/",
                target="_blank",
                rel="noopener noreferrer",
            ),
            ui.markdown(t("info", "license_text")),
        )
