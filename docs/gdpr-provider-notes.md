# GDPR / Data-Protection Provider Notes

*Research snapshot — May 2026. Hand to DPO for verification before contractual decisions. Background research for FEATURES.md items #8 (GDPR-compliant provider integration) and #9 (DPA / AVV document template).*

## 1. EU-hosted LLM providers (DACH/EU)

| Provider | API style | DPA / AVV | Pricing posture | Quality vs. GPT-4 class |
|---|---|---|---|---|
| **Aleph Alpha (PhariaAI)** | OpenAI-compatible REST, documented under `docs.aleph-alpha.com/phariaai-dev-guide` | Yes — AVV-fähig, hosted in DE; explicitly marketed for "sovereign AI" | Enterprise-tier; no public self-serve academic price list — **flag: needs sales contact** | Below GPT-4o on general benchmarks; competitive on German legal/admin text. **Flag:** April 2026 Aleph Alpha/Cohere merger talks — contractual continuity uncertain |
| **Mistral AI (La Plateforme)** | Native chat-completions API, OpenAI-shaped | Yes — DPA template at `legal.mistral.ai/terms/data-processing-addendum`; EU endpoints in Paris; 30-day abuse-retention, no training on API inputs by default | Pay-as-you-go; "Mistral Large 2" / "Medium 3" comparable to mid-tier OpenAI on price-per-1M tokens | Mistral Large 2 / Medium 3 ≈ GPT-4o-mini to GPT-4o on EU-language tasks; strong German |
| **IONOS AI Model Hub** | OpenAI-compatible REST (`api.ionos.com/docs/inference-openai/v1`) | Yes — AVV explicitly offered; servers in DE; ISO 27001 + BSI C5; no third-party sharing, no training on inputs | Token-based, no minimum; **flag:** prices visible only after Cloud Panel login | Hosts open-weight models (Llama 3.x, Mistral, Teuken-7B). Quality = whatever model you pick; no proprietary frontier tier |
| **Azure OpenAI (EU Data Zone)** | Standard OpenAI SDK with Azure endpoint | Microsoft DPA + EU Data Boundary; "Data Zone Standard (EUR)" keeps prompts/responses inside EU member states | Same per-token pricing as Azure OpenAI globally; EDU agreements via Microsoft volume licensing | Identical models to OpenAI direct (GPT-4o, GPT-4.1, o-series). **Note:** US parent → CLOUD Act exposure remains a topic for DPOs even with EU residency |
| **Schwarz Digits / STACKIT AI Model Serving** | OpenAI-compatible | DE/EU hosted; designed for sovereignty; AVV available via STACKIT contracting (standard cloud terms) | Token-based; positioned for enterprise — academic pricing **not publicly listed (flag)** | Hosts Llama 3.3 70B, Mistral Nemo, ~20B MXFP4 model. Quality = model-dependent. Will host the merged Aleph Alpha/Cohere stack |
| **OpenGPT-X / Teuken-7B** | Available via **IONOS AI Model Hub** and **Deutsche Telekom T-Systems AI Foundation Services** (unified API) | Via the host (IONOS AVV / T-Systems DPA) | Inference cost low (7B model) | Multilingual across 24 EU languages by design; **below** GPT-4 class — useful as cheap EU-sovereign baseline, not as primary coder. Project funding ended March 2025; weights remain on Hugging Face |

**Practical recommendation for TalkTrace AI neo:** Mistral La Plateforme and IONOS AI Model Hub are the two lowest-friction "Mittelweg" backends — both OpenAI-API-shaped (drop-in for the existing client), both with DE/EU hosting and signable AVV/DPA, both usable without enterprise sales calls. Azure OpenAI EU Data Zone is the right answer when an institution already has a Microsoft EA.

## 2. International compliance landscape

**UK (UK GDPR):** ICO has not banned US-hosted LLMs but treats transfers under the UK-US Data Bridge / SCCs. December 2024 ICO consultation response stresses transparency and lawful basis (typically legitimate interest); a statutory ADM code of practice is in progress. No UK-specific carve-out for education.

**USA (FERPA):** FERPA binds *schools/institutions*, not individual teacher use of personal tools. Pasting identifiable student speech into consumer ChatGPT is widely treated as a likely FERPA disclosure unless the vendor has signed a school agreement. OpenAI launched a FERPA-aligned "ChatGPT for Teachers" tier in 2025 with no-training-by-default and a student-data privacy agreement — relevant precedent for what schools expect.

**Canada:** PIPEDA federally; Ontario FIPPA (amended Nov 2024 by Bill 194 — public-sector AI disclosure + risk management). **BC FOIPPA effectively forbids storing public-sector personal data outside Canada** — a hard blocker for US-hosted backends in BC schools. Federal/Provincial Privacy Commissioners issued joint generative-AI principles in 2024.

**Australia:** Privacy Act 1988 + APPs. OAIC published two guidelines on 21 Oct 2024: one for *users* of commercial AI products, one for *developers* training models. Core message: do not enter personal/sensitive info into public GenAI; privacy-by-design across the lifecycle.

## 3. Self-hosted alternatives on a teacher's laptop

All figures Q4_K_M (4-bit) via llama.cpp / Ollama; tok/s rough.

| Model | RAM (Q4) | M2/M3 MacBook | Mid-range Win laptop (CPU/iGPU) | Quality for DE qualitative coding |
|---|---|---|---|---|
| **Llama 3.1 8B** | ~5–6 GB | 27–48 tok/s (M2 Pro, llama.cpp Metal) | 5–12 tok/s CPU; 15–25 with decent dGPU | Solid English; usable German with prompt-tuning. Good baseline |
| **Mistral 7B / Mistral Small 3 (24B)** | 7B ≈ 5 GB; Small ≈ 14–16 GB | 7B: 35–50 tok/s; Small: ~10–18 tok/s on M3 Max | 7B: 6–14 tok/s; Small: only on 32 GB+ machines | 7B mediocre for nuanced coding; Small 3 noticeably better, near GPT-4o-mini for structured tasks |
| **Qwen 2.5 7B** | ~5–6 GB | Comparable to Llama 3.1 8B | Comparable | Outperforms Llama 3.1 8B and Gemma 2 9B on most general benchmarks; German weaker than English. **Flag:** Chinese provenance may matter for some DPOs |
| **Gemma 2 9B** | ~6–7 GB | 25–40 tok/s | 4–10 tok/s CPU | Strong instruction-following; decent German; output style sometimes verbose for coding tasks |

**Realistic floor:** A teacher with 16 GB RAM can run an 8–9B model at usable speed. Anything ≥ ~14B (Mistral Small, Gemma 2 27B) needs 32 GB or a discrete GPU and is not a mass-deployment story.

## Open flags for the DPO

- Aleph Alpha/Cohere merger — wait-and-see before signing a multi-year AVV.
- IONOS and STACKIT pricing not on the public web; need quotes.
- Azure OpenAI EU Data Zone vs. CLOUD Act exposure — institution-specific risk call.
- No public benchmark located for "German qualitative coding" specifically; recommend an internal eval on a fixture transcript before locking in any backend.

## Sources

- [Mistral Data Processing Addendum](https://legal.mistral.ai/terms/data-processing-addendum)
- [Mistral Help — Where can I consult your DPA](https://help.mistral.ai/en/articles/347641-where-can-i-consult-your-dpa-data-processing-agreement)
- [Aleph Alpha — Sovereign AI](https://aleph-alpha.com/)
- [Aleph Alpha API reference](https://docs.aleph-alpha.com/phariaai-dev-guide/latest/pharia-openapi/index.html)
- [IONOS AI Model Hub](https://cloud.ionos.com/managed/ai-model-hub)
- [IONOS OpenAI-compatible API docs](https://api.ionos.com/docs/inference-openai/v1/)
- [IONOS — Teuken 7B](https://docs.ionos.com/cloud/ai/ai-model-hub/models/llms/opengpt-x-teuken)
- [Azure OpenAI Data Zones announcement](https://azure.microsoft.com/en-us/blog/announcing-the-availability-of-azure-openai-data-zones-and-latest-updates-from-azure-ai/)
- [Microsoft EU Data Boundary](https://learn.microsoft.com/en-us/privacy/eudb/eu-data-boundary-learn)
- [STACKIT AI Model Serving](https://stackit.com/en/products/data-ai/stackit-ai-model-serving)
- [STACKIT available shared models](https://docs.stackit.cloud/products/data-and-ai/ai-model-serving/basics/available-shared-models/)
- [OpenGPT-X / Teuken-7B (Fraunhofer IAIS)](https://www.iais.fraunhofer.de/en/industries_and_cross-sector_solutions/cross-sector_solutions/generative-ai/opengpt-x.html)
- [Telekom — Teuken commercial offering](https://www.telekom.com/en/media/media-information/archive/opengpt-x-language-model-made-in-germany-1084484)
- [ICO — Guidance on AI and data protection](https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/guidance-on-ai-and-data-protection/)
- [Osborne Clarke — ICO updated views on GenAI](https://www.osborneclarke.com/insights/ico-updates-its-views-using-personal-data-generative-ai-uk)
- [OpenAI — ChatGPT for Teachers (FERPA)](https://openai.com/index/chatgpt-for-teachers/)
- [Future of Privacy Forum — Vetting GenAI for schools](https://fpf.org/wp-content/uploads/2024/10/Ed_AI_legal_compliance.pdf_FInal_OCT24.pdf)
- [HillNotes — Privacy and AI in Canada](https://hillnotes.ca/2025/05/27/privacy-and-artificial-intelligence-in-canada/)
- [UBC — PIA Guidelines for GenAI (July 2025)](https://privacymatters.ubc.ca/sites/default/files/2025-07/PIA_Guidelines_GenAI_2025-07.pdf)
- [OAIC — Guidance on commercially available AI products](https://www.oaic.gov.au/privacy/privacy-guidance-for-organisations-and-government-agencies/guidance-on-privacy-and-the-use-of-commercially-available-ai-products)
- [OAIC — Developing/training generative AI models](https://www.oaic.gov.au/privacy/privacy-guidance-for-organisations-and-government-agencies/guidance-on-privacy-and-developing-and-training-generative-ai-models)
- [llama.cpp Apple Silicon performance discussion](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [Apple Silicon LLM benchmarks (llmcheck.net)](https://llmcheck.net/benchmarks)
- [Qwen 2.5 LLM blog](https://qwenlm.github.io/blog/qwen2.5-llm/)
