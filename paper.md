---
title: "TalkTrace-AI: An LLM-Based Learning Analytics Tool for Analyzing Classroom Dialogue in Teacher Education and Teacher Professional Development"
tags:
  - teacher education
  - teacher professional development
  - classroom dialogue
  - learning analytics
  - large language models
  - educational software
authors:
  - name: Dennis Hauk
    orcid: 0000-0002-5779-2876
    affiliation: 1
  - name: Jan-Michael Schorling
    orcid: 0009-0005-9007-2896
    affiliation: 1
affiliations:
  - name: Chair for Teaching and Learning in Civic Education, Institute of Political Science, Faculty of Social Sciences and Philosophy, Leipzig University, Leipzig, Germany
    index: 1
date: 2026-04-17
bibliography: paper.bib
---

# Summary

TalkTrace-AI is a browser-based, open-source web application for analyzing classroom dialogue from transcripts. The tool was developed for teacher education and teacher professional development, with a particular focus on settings in which instructors and participants need timely, structured feedback on classroom talk. TalkTrace-AI combines two analytic layers. First, it produces quantitative indicators of participation and turn distribution, such as number of turns, mean turn length, teacher talk share, and participation rate. Second, it supports qualitative coding of teacher talk moves through a user-defined codebook and large language models.

The tool is designed to make discourse analysis more accessible in educational settings where full manual coding is often too time-consuming. At the same time, it aims to remain transparent and configurable. Users can upload their own codebooks, select the model, inspect and adapt prompts, and export or re-import sessions. TalkTrace-AI is therefore not a fixed scoring system but a configurable analytic workflow for structured reflection on teaching.

The intended users are teacher educators, teacher professional development facilitators, and individual teachers who want to examine discussion-based instruction with manageable technical effort. In its current use, TalkTrace-AI supports reflection on leading classroom discussion in pre-service teacher education.

# Statement of Need

Classroom dialogue is widely regarded as a central medium of learning, reasoning, and participation. At the same time, research has repeatedly shown that classroom talk often remains teacher-dominated and that more dialogic forms of interaction are difficult to establish and sustain [@HoweAbedin2013; @Alexander2020]. For teacher education and professional development, this creates a practical challenge. If classroom dialogue is to become an object of professional learning, teachers need usable ways to examine interaction patterns in their own or simulated teaching.

Existing research offers coding frameworks and analytic indicators for this purpose [@TaoChen2023; @HennessyEtAl2020]. However, applying such frameworks usually requires substantial manual work. Transcripts must be prepared, turns segmented, categories assigned, and results summarized for feedback. This limits scalability and makes it difficult to provide timely evidence for seminar-based reflection or professional development workshops [@HennessyEtAl2020]. Recent work on AI-supported discourse analysis suggests that automated or semi-automated tools can reduce this burden and support teacher learning when their outputs are interpretable and pedagogically meaningful [@Chen2020; @JacobsEtAl2022; @WangChen2024].

TalkTrace-AI addresses this need as open-source educational software. It was developed from a concrete teacher-education workflow in which facilitators needed structured dialogue indicators from micro-teaching transcripts without building a full manual coding pipeline for each seminar group. Its contribution lies in making existing analytic approaches more usable for teaching and learning contexts.

# Software Description

TalkTrace-AI analyzes transcripts of classroom or micro-teaching episodes. The expected format is simple: the teacher is marked with a fixed label such as `TEACHER`, while students are identified with anonymized labels such as `S01`, `S02`, and `S03`. Users may also enter contextual metadata such as group ID, class size, and number of participating students.

The software combines a quantitative and a qualitative layer. The quantitative layer computes indicators that describe participation and distributional aspects of the conversation, including number of turns, mean turn length, word distribution across teacher and students, participation rate, and teacher talk share. These outputs provide a compact overview of whether talk is broadly distributed or concentrated and whether teacher contributions dominate the interaction.

The qualitative layer focuses on teacher impulses. Users upload a codebook that defines the categories to be applied, for example invite ideas, expand ideas, challenge, invitation for reasoning, guide, or connect contributions. TalkTrace-AI then uses a large language model to assign codebook categories to teacher turns. Because the categories are user-defined, the tool is not tied to one pedagogical framework or subject area. It can be adapted to different coding schemes and local instructional purposes.

The interface is organized into three tabs: `Analysis`, `Results`, and `Options`. In `Analysis`, users enter metadata, upload transcript and codebook files, inspect previews, and start the analysis. In `Results`, the software displays quantitative summaries, qualitative code distributions, and a structured overview of coded teacher impulses, together with downloadable reports. In `Options`, users can configure the API key, select the model, inspect or revise prompts, and control additional settings.

A key design feature is transparent configuration. Prompts, codebooks, and model choices are visible rather than hidden. This supports critical inspection and makes the analytic setup easier to document and adapt.

Privacy-sensitive processing is also central to the architecture. TalkTrace-AI itself does not store transcripts or analysis results on an external server beyond the active session. Preparatory and display-related processing takes place locally in the browser session. Only the LLM-based qualitative coding step requires online API communication. When this step is activated, the relevant transcript segments and the uploaded codebook are transmitted to the selected provider. Any provider-side retention or logging therefore depends on that provider’s policies. API keys are stored in the operating system’s encrypted password store, and session data can be exported and re-imported locally.

# Usage in Teaching and Learning

TalkTrace-AI is currently used at Leipzig University in a teacher-education context focused on civic education. There, it is embedded in a core-practice-oriented program that addresses leading classroom discussion as a central teaching practice. The instructional sequence combines theoretical input on dialogic teaching, modelling and rehearsal, short micro-teaching simulations, automated transcription, and structured reflection.

In a typical cycle, pre-service teachers conduct a discussion-based micro-teaching sequence of about ten minutes. The session is audio- and video-recorded, then transcribed with noScribe [@noScribe]. After anonymization and formatting, the transcript is analyzed with TalkTrace-AI. In the subsequent seminar session, the output is used together with selected video excerpts for guided reflection.

This use is not intended as summative assessment. Instead, the software functions as a structured reflective resource. Quantitative indicators help participants notice broad participation patterns, for example limited student participation or a high teacher talk share. The qualitative coding layer supports closer examination of teacher talk moves, such as whether a discussion relied mainly on guiding moves or whether it also included invitations to elaborate, reason, or connect contributions.

In this way, the tool helps teacher educators move more efficiently from recorded teaching to evidence-informed reflection. Recent course use indicates that the workflow is feasible for regular seminar settings and that the outputs can be meaningfully integrated into guided reflection on classroom dialogue.

# Availability and Reuse

The source code for TalkTrace-AI is available in a public repository: [TalkTrace repository](https://github.com/talktrace-ai/talktrace-ai). A downloadable source archive is currently available via GitHub: [GitHub source archive](https://github.com/talktrace-ai/talktrace-ai/archive/refs/heads/main.zip). The repository includes the software source code, installation guidance, and license information.

Because the software is browser-based and codebook-driven, it can be adopted without building a new analytic pipeline from scratch. Reuse is not limited to civic education. The underlying workflow can be adapted to other subjects and professional-development settings in which discussion-based teaching matters. The main requirements are anonymized transcripts, a suitable codebook, and an appropriate local decision about which LLM provider to use. This makes TalkTrace-AI particularly suited to contexts that need configurable, inspectable, and reusable software for reflection on instructional dialogue.
