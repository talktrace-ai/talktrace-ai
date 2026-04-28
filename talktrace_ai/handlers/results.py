"""Results section: quantitative + qualitative output panels."""
from ._common import *

from ..utils.codebook_hierarchy import build_priority_lookup, priority_for


def _is_multi_coding_on(input_obj) -> bool:
    """Read the sidebar switch defensively. The switch is only rendered when
    LLM analysis is active, so before that input.multi_coding_switch() raises.
    Default: False (single-coding, hierarchy-resolved)."""
    try:
        return bool(input_obj.multi_coding_switch())
    except Exception:
        return False


def register(state):
    input = state.input
    output = state.output
    session = state.session
    t = state.t
    transcript_data = state.transcript_data
    codebook_data = state.codebook_data
    converted_transcript = state.converted_transcript
    num_participants = state.num_participants
    participation_rate = state.participation_rate
    stats = state.stats
    stats_per_speaker = state.stats_per_speaker
    llm_analysis_data = state.llm_analysis_data
    analysis_state = state.analysis_state
    analysis_llm_state = state.analysis_llm_state
    sim_plot = state.sim_plot
    qual_plot = state.qual_plot
    qual_stats_df = state.qual_stats_df
    placeholder_plot = state.placeholder_plot
    code_legend_storage = state.code_legend_storage

    ### Ergebnisse --------------------------------------------------------
    # Ergebnisse Tab Titel
    @render.text
    def loc_title_results():
        return (t("results", "tab_title"))


    # Warnung, wenn Ergebnisse Tab ohne Analyse angeklickt wird
    @reactive.effect
    @reactive.event(input.main_tabs)
    def warn_if_results_tab_clicked():
        if input.main_tabs() == '<div id="loc_title_results" class="shiny-text-output"></div>' and not analysis_state.get():
            m = ui.modal(
                ui.p(t("results", "no_results")),
                ui.tags.hr(),
                ui.p(t("onboarding", "empty_results_message")),
                ui.input_action_button(
                    "tt_demo_open_from_modal",
                    t("onboarding", "demo_button"),
                    icon=icon_svg("vial"),
                    class_="btn-primary btn-sm",
                ),
                title=t("results", "no_results_title"),
                easy_close=True,
                footer=ui.modal_button("OK", class_="btn-success"),
                size="m",
            )
            ui.modal_show(m)
            ui.update_navs("main_tabs", selected='<div id="loc_title_analysis" class="shiny-text-output"></div>')

    # Anzeige der allgemeinen Informationen
    @render.ui
    def loc_quantitative_analysis():
        return ui.span(t("results", "section_quantitative_analysis"))


    # Die Berechnung der Stats-Werte (t_turns, p_turns, ...) erfolgt jetzt
    # direkt in run_analysis(), damit sie auch ohne gerenderten Results-Tab
    # für Report-Download und Session-Export verfügbar sind.


    # Anzeige der Gruppen-ID
    @render.ui
    def loc_group_id_display():
        return ui.value_box(
                        ui.p(t("analysis", "group_id")),
                        ui.output_text("nameGroup"),
                        showcase=icon_svg("id-card"),
                    ),


    @render.text
    def nameGroup():
        return input.name_group()


    # Anzeige der Klassengröße
    @render.ui
    def loc_class_size():
        return ui.value_box(
                        ui.p(t("results", "class_size")),
                        ui.output_text("numPupils"),
                        showcase=icon_svg("user-group"),
                    ),


    @render.text
    def numPupils():
        return input.num_pupils()


    # Anzeige der Anzahl beteiligter Schüler:innen
    @render.ui
    def loc_num_participants():
        return ui.value_box(
                        ui.p(t("results", "num_participants")),
                        ui.output_text("numParticipants"),
                        showcase=icon_svg("user-check"),
                    ),


    @render.text
    def numParticipants():
        req(num_participants.get() != None)
        return num_participants.get()


    # Anzeige der Beteiligungsquote
    @render.ui
    def loc_participation_rate():
        return ui.value_box(
                        ui.p(t("results", "participation_rate")),
                        ui.output_text("participationRate"),
                        showcase=icon_svg("square-poll-vertical"),
                    ),


    @render.text
    def participationRate():
        req(participation_rate.get() is not None)
        return f"{round(participation_rate.get(), 2)} %"


    # Verteilung der Gesprächsbeiträge
    @render.ui
    def loc_distribution_of_turns():
        return ui.p(t("results", "distribution_of_turns"))


     # Create a bar plot for quantitative statistics
    @reactive.calc
    def make_sim_stats_plot():
        req(transcript_data.get() != None)

        stats_df = stats.get()
        distribution = stats_df.plot(kind='bar', x='Sprecher', y='Gesamt_Woerter', alpha=1, rot=0)
        distribution.set_xlabel(t("results", "words_total"))
        distribution.set_ylabel(t("results", "quantity"))
        distribution.set_axisbelow(True)
        distribution.grid(color='gray', axis='y')
        legend = distribution.get_legend()
        if legend is not None:
            legend.remove()
        total = stats_df['Gesamt_Woerter'].sum() or 1  # avoid div-by-zero when empty
        # Build tick labels matching whatever rows are actually present in stats_df.
        # Map each speaker row to teacher or students using the user-provided name.
        teacher_label = t("stats", "teacher")
        students_label = t("stats", "students")
        teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
        tick_labels = [
            teacher_label if str(spk) == teacher_name else students_label
            for spk in stats_df['Sprecher'].tolist()
        ]
        distribution.set_xticks(range(len(tick_labels)))
        distribution.set_xticklabels(tick_labels)
        for container in distribution.containers:
            perc_labels = [f"{(bar.get_height() / total * 100):.1f}%" for bar in container]

            distribution.bar_label(container, label_type='center')
            distribution.bar_label(container, labels=perc_labels, label_type='edge')

        sim_plot.set(distribution)
        return distribution


    # Plot für Gesprächsverteilung
    @render.plot(alt="placeholder", height=260)
    def sim_stats_plot():
        if analysis_state.get() == False:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        else:
            return make_sim_stats_plot()


    @render.ui
    def loc_over_time_quant_title():
        return ui.span(t("results", "over_time_quant_title"))


    def _segment_labels_for(n_segments):
        if n_segments == 3:
            return [t("results", "section_first"),
                    t("results", "section_middle"),
                    t("results", "section_last")]
        return [f"{t('results', 'section')} {i + 1}" for i in range(n_segments)]

    state.segment_labels_for = _segment_labels_for


    @reactive.calc
    def make_sim_stats_over_time_plot():
        req(transcript_data.get() is not None)
        transcript = transcript_data.get()
        teacher = input.name_teacher() or t("analysis", "name_teacher_var")
        n_segments = 3
        df = dialog_stats_over_time(
            transcript, teacher,
            n_segments=n_segments,
            segment_labels=_segment_labels_for(n_segments),
        )
        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        pivot = df.pivot(index="Abschnitt", columns="Sprecher_Gruppe", values="Wörter") \
                  .reindex(_segment_labels_for(n_segments))
        ax = pivot.plot(kind='bar', rot=0, alpha=1)
        ax.set_xlabel(t("results", "section"))
        ax.set_ylabel(t("results", "words_total"))
        ax.set_axisbelow(True)
        ax.grid(color='gray', axis='y')
        ax.legend(loc="upper right", fontsize=8, title=None)
        for container in ax.containers:
            ax.bar_label(container, label_type='edge', fontsize=8)
        return ax.get_figure()

    state.make_sim_stats_over_time_plot = make_sim_stats_over_time_plot


    @render.plot(alt="placeholder", height=240)
    def sim_stats_over_time_plot():
        if not analysis_state.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        return make_sim_stats_over_time_plot()

    # Gesprächsstatistiken
    @render.ui
    def loc_interaction_turns():
        return ui.p(t("results", "interaction_turns"))

    # Lehrperson
    @render.ui
    def loc_teacher():
        return ui.p(t("results", "teacher"))


    @render.ui
    def loc_teacher_turns():
        return ui.markdown(f"**{t("results", "turn_count")}**")


    # Gesprächsbeiträge Lehrperson
    @render.text
    def teacher_turns():
        req(analysis_state.get(), transcript_data.get() != None)
        m = stats.get().loc[stats.get()['Sprecher'] == input.name_teacher(), 'Anzahl_Beitraege']
        return m.values[0] if not m.empty else 0


    # Display the TOTAL number of turns (teacher + all students).
    @render.text
    def teacher_impulses():
        req(stats.get() is not None and not stats.get().empty)
        total_turns = int(stats.get()['Anzahl_Beitraege'].sum())
        return total_turns


    @render.ui
    def loc_teacher_turns_length():
        return ui.markdown(f"**{t("results", "turn_length")}**")


    # Länge der Gesprächsbeiträge Lehrperson
    @render.text
    def teacher_turns_length():
        req(analysis_state.get(), transcript_data.get() != None)
        df = stats.get()
        teacher = input.name_teacher()
        avg = df.loc[df['Sprecher'] == teacher, 'Durchschnitt_Woerter']
        med = df.loc[df['Sprecher'] == teacher, 'Median_Woerter']
        return f"{round(avg.values[0], 1) if not avg.empty else 0} ({round(med.values[0], 1) if not med.empty else 0})"


    # Schüler:innen
    @render.ui
    def loc_pupils():
        return ui.p(t("results", "students"))


    @render.ui
    def loc_pupils_turns():
        return ui.markdown(f"**{t("results", "turn_count")}**")


    # Gesprächsbeiträge Schüler:innen
    @render.text
    def pupils_turns():
        req(analysis_state.get(), transcript_data.get() != None)
        m = stats.get().loc[stats.get()['Sprecher'] == "Schüler:innen", 'Anzahl_Beitraege']
        return m.values[0] if not m.empty else 0


    @render.ui
    def loc_pupils_turns_length():
        return ui.markdown(f"**{t("results", "turn_length")}**")


    # Länge der Gesprächsbeiträge Schüler:innen
    @render.text
    def pupils_turns_length():
        req(analysis_state.get(), transcript_data.get() != None)
        df = stats.get()
        avg = df.loc[df['Sprecher'] == "Schüler:innen", 'Durchschnitt_Woerter']
        med = df.loc[df['Sprecher'] == "Schüler:innen", 'Median_Woerter']
        return f"{round(avg.values[0], 1) if not avg.empty else 0} ({med.values[0] if not med.empty else 0})"


    # Anzeige der Qualitativen Analyse
    @render.ui
    def loc_qualitative_analysis():
        return ui.span(t("results", "section_qualitative_analysis"))


    # Quick Stats
    @render.ui
    def loc_impulses_count():
        return ui.p(t("results", "impulses_count"))


    @render.ui
    def loc_coded_impulses():
        return ui.p(t("results", "coded_impulses"))


    # Display the number of impulses coded
    @render.text
    def teacher_impulses_coded():
        req(analysis_llm_state.get(), analysis_state.get())
        df = qual_stats_df.get()
        if df is None or df.empty:
            return "0"
        # Exclude uncoded turns (empty Shortcode from the LEFT JOIN in
        # make_qualitative_stats_df) — same filter as code_most_used.
        codes = df[t("report", "shortcode")].astype(str).str.strip()
        return int((codes != "").sum())


    @render.ui
    def loc_most_frequent_codes():
        return ui.p(t("results", "most_frequent_codes"))

    # Display the most used code
    @render.text
    def code_most_used():
        req(analysis_llm_state.get(), analysis_state.get())
        # Find the most used code
        try:
            df = qual_stats_df.get()
            if df is None or df.empty:
                return t("system_prompts", "no_code")
            # Exclude uncoded turns (empty Shortcode from the LEFT JOIN in
            # make_qualitative_stats_df) — otherwise "" usually wins the mode().
            codes = df[t("report", "shortcode")].astype(str).str.strip()
            codes = codes[codes != ""]
            if codes.empty:
                return t("system_prompts", "no_code")
            most_used_codes = codes.mode().to_list()
            return ', '.join(most_used_codes) if most_used_codes else t("system_prompts", "no_code")
        except Exception:
            return t("system_prompts", "no_code")


    @render.ui
    def loc_teacher_talking_rate():
        return ui.p(t("results", "teacher_talking_rate"))


    # Display the share of words spoken by teacher vs. students,
    # plus an expandable popover with a per-student breakdown.
    @render.ui
    def teacher_share_ui():
        req(stats.get() is not None and not stats.get().empty)
        df = stats.get()
        total_words = df['Gesamt_Woerter'].sum()
        teacher_name = input.name_teacher()

        tw = df.loc[df['Sprecher'] == teacher_name, 'Gesamt_Woerter']
        teacher_words = tw.values[0] if not tw.empty else 0

        sw = df.loc[df['Sprecher'] == "Schüler:innen", 'Gesamt_Woerter']
        student_words = sw.values[0] if not sw.empty else 0

        if total_words <= 0:
            return ui.span("0 %")

        t_share = round(teacher_words / total_words * 100, 1)
        s_share = round(student_words / total_words * 100, 1)

        teacher_label = t("results", "teacher")
        students_label = t("results", "students")

        # Per-student breakdown for the popover.
        per_speaker_df = stats_per_speaker.get()
        details_rows = []
        if per_speaker_df is not None and not per_speaker_df.empty:
            # Teacher row first.
            t_row = per_speaker_df.loc[per_speaker_df['Sprecher'] == teacher_name]
            if not t_row.empty:
                w = int(t_row['Gesamt_Woerter'].values[0])
                pct = round(w / total_words * 100, 1) if total_words > 0 else 0
                details_rows.append((teacher_label, w, pct))
            # Each student, sorted by speaker label (S01, S02, ...).
            student_rows = per_speaker_df.loc[per_speaker_df['Sprecher'] != teacher_name].sort_values('Sprecher')
            for _, r in student_rows.iterrows():
                w = int(r['Gesamt_Woerter'])
                pct = round(w / total_words * 100, 1) if total_words > 0 else 0
                details_rows.append((str(r['Sprecher']), w, pct))

        # Build the popover body: a compact, scrollable table.
        table_rows = [
            ui.tags.tr(
                ui.tags.th(t("results", "speaker"), style="text-align:left; padding:2px 8px;"),
                ui.tags.th(t("results", "words_total"), style="text-align:right; padding:2px 8px;"),
                ui.tags.th("%", style="text-align:right; padding:2px 8px;"),
            )
        ]
        for label, w, pct in details_rows:
            table_rows.append(
                ui.tags.tr(
                    ui.tags.td(label, style="text-align:left; padding:2px 8px;"),
                    ui.tags.td(f"{w}", style="text-align:right; padding:2px 8px;"),
                    ui.tags.td(f"{pct} %", style="text-align:right; padding:2px 8px;"),
                )
            )
        details_table = ui.tags.div(
            ui.tags.table(*table_rows, style="font-size:0.85rem; border-collapse:collapse; width:100%;"),
            style="max-height:300px; overflow-y:auto;",
        )

        summary = ui.tags.span(
            f"{teacher_label}: {t_share} % | {students_label}: {s_share} %",
            style="font-size:0.95rem;",
        )
        details_btn = ui.tags.span(
            ui.popover(
                ui.tags.a("Details ▾", href="#", style="font-size:0.8rem; margin-left:0.5rem; text-decoration:underline; cursor:pointer;"),
                details_table,
                title=t("results", "teacher_talking_rate"),
                placement="bottom",
            )
        )
        return ui.tags.div(summary, details_btn)


    # Qualitative Statistics Plot for Coded Impulses
    @render.ui
    def loc_impulses_distribution():
        ui.p(t("results", "impulses_distribution"))


    # Create a bar plot for qualitative statistics
    @reactive.calc
    def make_qualitative_stats_plot():
        req(llm_analysis_data.get())
        # Reuse the merged DataFrame from make_qualitative_stats_df so the
        # bar plot stays consistent with the table: same hierarchy resolution,
        # same multi-coding aggregation. With multi-coding ON cells contain
        # "RE; A; CO" which we split + explode below so each code is counted
        # individually.
        merged_df = make_qualitative_stats_df()
        if merged_df is None or merged_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            qual_plot.set(ax)
            return ax
        shortcode_col = t("report", "shortcode")
        plot_df = merged_df.copy()
        plot_df[shortcode_col] = plot_df[shortcode_col].astype(str).str.strip()
        # Split multi-coded cells. For single-coding cells the regex returns
        # a single-element list, so explode is a no-op.
        plot_df[shortcode_col] = plot_df[shortcode_col].str.split(r"\s*;\s*", regex=True)
        plot_df = plot_df.explode(shortcode_col)
        plot_df[shortcode_col] = plot_df[shortcode_col].astype(str).str.strip()
        plot_df = plot_df[plot_df[shortcode_col] != ""]
        if plot_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            qual_plot.set(ax)
            return ax
        analysis_plot = plot_df.groupby(shortcode_col).agg(
            Anzahl=(shortcode_col, 'count'),
            ).reset_index().plot(kind='bar', x=shortcode_col, y='Anzahl', alpha=1, rot=0)
        analysis_plot.set_xlabel(t("report", "shortcode"))
        # Rotate tick labels without resetting ticks (avoids FixedLocator/labels mismatch)
        plt.setp(analysis_plot.get_xticklabels(), rotation=45, ha='right')
        analysis_plot.set_ylabel(t("report", "quantity"))
        analysis_plot.set_axisbelow(True)
        analysis_plot.grid(color='gray', axis = 'y')
        legend = analysis_plot.get_legend()
        if legend is not None:
            legend.remove()
        for container in analysis_plot.containers:
            analysis_plot.bar_label(container, label_type='edge')
        qual_plot.set(analysis_plot)
        return analysis_plot


    # Plot für qualitative Statistik
    @render.plot(alt="Noch keine Daten", height=260)
    def qualitative_stats_plot():
        if not llm_analysis_data.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        else:
            return make_qualitative_stats_plot()


    @render.ui
    def loc_over_time_quali_title():
        return ui.span(t("results", "over_time_quali_title"))


    @reactive.calc
    def make_qualitative_stats_over_time_plot():
        req(llm_analysis_data.get())
        req(transcript_data.get() is not None)
        latest_df = llm_analysis_data.get()[-1]
        if latest_df is None or latest_df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        transcript = transcript_data.get()
        teacher = input.name_teacher() or t("analysis", "name_teacher_var")
        n_segments = 3
        labels = _segment_labels_for(n_segments)
        mapped = map_impulses_to_turn_index(latest_df, transcript, teacher)
        total_turns = count_transcript_turns(transcript, teacher)
        dist = code_distribution_over_time(
            mapped, total_turns,
            n_segments=n_segments,
            segment_labels=labels,
        )
        if dist.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        pivot = (dist.pivot(index="Abschnitt", columns="Shortcode", values="Anteil")
                     .fillna(0)
                     .reindex(labels))
        ax = pivot.plot(kind='bar', stacked=True, rot=0, alpha=1)
        ax.set_xlabel(t("results", "section"))
        ax.set_ylabel(t("results", "share"))
        ax.set_ylim(0, 1)
        ax.set_axisbelow(True)
        ax.grid(color='gray', axis='y')
        ax.legend(loc="upper right", fontsize=8, title=t("report", "shortcode"),
                  bbox_to_anchor=(1.0, 1.0))
        return ax.get_figure()

    state.make_qualitative_stats_over_time_plot = make_qualitative_stats_over_time_plot


    @render.plot(alt="placeholder", height=240)
    def qualitative_stats_over_time_plot():
        if not llm_analysis_data.get():
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
            ax.axis('off')
            return fig
        return make_qualitative_stats_over_time_plot()


    # DataFrame of Coded Impulses
    @render.ui
    def loc_impulses_coding():
        ui.p(t("results", "impulses_coding"))


    # Create a DataFrame for qualitative statistics
    @reactive.calc
    def make_qualitative_stats_df():
        req(llm_analysis_data.get())
        analysis_df = llm_analysis_data.get()[-1].copy()
        cols = ['#', t("report", "speaker"), t("report", "teacher_statement"), t("report", "shortcode")]
        # Empty analysis (no codable turns) -> return empty, properly-named df
        if analysis_df.empty:
            empty_df = pd.DataFrame(columns=cols)
            qual_stats_df.set(empty_df)
            return empty_df
        # Back-fill Sprecher column if missing (older sessions)
        if "Sprecher" not in analysis_df.columns:
            analysis_df["Sprecher"] = ""

        transcript_text = transcript_data.get()
        if not transcript_text:
            # Fallback: converted transcript (format wizard result)
            conv = converted_transcript.get()
            if conv and conv.get("text"):
                transcript_text = conv["text"]
        if transcript_text:
            teacher_name = input.name_teacher() or t("analysis", "name_teacher_var")
            turns = _parse_turns(transcript_text, teacher_name)
            all_turns_df = pd.DataFrame(turns, columns=["Sprecher", "Impuls"])
            # Normalize parsed speaker to canonical teacher_name (case-insensitive
            # regex may produce the verbatim transcript casing, e.g. "Lehrer" vs "LEHRER").
            all_turns_df["Sprecher"] = all_turns_df["Sprecher"].apply(
                lambda s: teacher_name if s.lower() == teacher_name.lower() else s
            )
            all_turns_df['#'] = range(1, len(all_turns_df) + 1)
            # Build a tolerant merge key: lowercase, strip surrounding punctuation
            # and collapse internal whitespace. LLMs frequently return Impulse
            # text with minor edits (trimmed trailing periods, normalized
            # quotes, collapsed whitespace) — exact-match on the raw string
            # would leave every Shortcode cell empty for real-LLM runs.
            def _norm_impuls(s):
                t_ = re.sub(r"\s+", " ", str(s)).strip()
                return re.sub(r"^[\s\"'„“”»«()\[\]\.…!?,:;-]+|[\s\"'„“”»«()\[\]\.…!?,:;-]+$", "", t_).lower()
            all_turns_df["__key__"] = all_turns_df["Sprecher"] + " :: " + all_turns_df["Impuls"].apply(_norm_impuls)
            coded = analysis_df[["Sprecher", "Impuls", "Shortcode"]].copy()
            # Normalize teacher speaker name: LLMs sometimes return "Lehrperson" or
            # "Lehrer" even when the transcript uses the configured teacher_name (e.g.
            # "LEHRER"). Map any case-insensitive match to the canonical name so the
            # join key aligns with all_turns_df.
            _teacher_aliases = {"lehrperson", "lehrer", "lehrkraft", "teacher", teacher_name.lower()}
            coded["Sprecher"] = coded["Sprecher"].apply(
                lambda s: teacher_name if str(s).lower() in _teacher_aliases else s
            )
            coded["__key__"] = coded["Sprecher"] + " :: " + coded["Impuls"].apply(_norm_impuls)
            # Hierarchie aus dem Codebuch ableiten (Position oder explizite
            # Priorität-Spalte). Codes ausserhalb des Codebuchs landen ans Ende.
            _priority_lookup = build_priority_lookup(codebook_data.get())
            coded["__priority__"] = coded["Shortcode"].apply(
                lambda c: priority_for(_priority_lookup, str(c).strip())
            )
            # Stabiler Sort: nach Priorität (aufsteigend = höhere Priorität zuerst).
            coded = coded.sort_values("__priority__", kind="mergesort")
            if _is_multi_coding_on(input):
                # Mehrfach-Codierung: Codes pro Turn in Priorität-Reihenfolge
                # mit "; " verbinden. Doppelte Codes pro Turn werden dedupliziert
                # (dict.fromkeys behält Reihenfolge).
                coded = (
                    coded.groupby("__key__", sort=False)
                         .agg({"Shortcode": lambda s: "; ".join(dict.fromkeys(str(c).strip() for c in s if str(c).strip()))})
                         .reset_index()
                )
            else:
                # Single-Coding: höchstpriore Code überlebt pro Turn.
                coded = coded.drop_duplicates(subset=["__key__"], keep="first")
                coded = coded.drop(columns=["__priority__"])
            merged = pd.merge(
                all_turns_df,
                coded[["__key__", "Shortcode"]],
                on="__key__",
                how="left",
            )
            merged = merged.drop(columns=["__key__"])
            merged = merged[['#', 'Sprecher', 'Impuls', 'Shortcode']].copy()
            merged["Shortcode"] = merged["Shortcode"].fillna("").astype(str)
            merged.columns = cols
            qual_stats_df.set(merged)
            return merged
        else:
            # Fallback: just coded impulses (no transcript available)
            analysis_df['#'] = analysis_df.reset_index().index + 1
            analysis_df = analysis_df[['#', "Sprecher", "Impuls", "Shortcode"]]
            analysis_df.columns = cols
            qual_stats_df.set(analysis_df)
            return analysis_df


# DataFrame für qualitative Statistik generieren
    @render.table()
    def qualitative_stats_df():
        return make_qualitative_stats_df()


    # DataFrame für qualitative Statistik
    @render.ui
    def quali_stats_df():
        if not llm_analysis_data.get():
            return ui.output_plot("placeholder")
        else:
            return ui.output_table("qualitative_stats_df")


    # Placeholder Plot, wenn noch keine Daten vorhanden sind
    @render.plot(alt="Noch keine Daten")
    def placeholder():
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, t("results", "no_data"), ha='center', va='center', fontsize=12)
        ax.axis('off')
        placeholder_plot.set(fig)
        return fig


    @render.plot(alt="Noch keine Daten")
    def placeholder2():
        return placeholder_plot.get()


    # Code-Legende aus Codebuch extrahieren
    @reactive.effect
    def extract_code_legend():
        data = codebook_data.get()
        req(data != None)
        if isinstance(data, list):
            df = pd.DataFrame(data)
            legend = [f"{code}" for code in df[df.columns[0]].unique()]
            code_legend_storage.set("; ".join(legend))
        else:
            code_legend_storage.set(str(data))


    # Code-Legende anzeigen
    @render.ui
    def code_legend():
        return ui.markdown(f"**{t("results", "caption")}:** {code_legend_storage.get()}")
