# Downloaded Papers

25 PDFs plus one blog post (HTML). `text/` holds pypdf text extractions (pages delimited by
`=== PAGE n ===`); `notes/` holds detailed per-paper reading notes for the ten user-specified
sources (prompts, probe recipes, numbers with page references).

## User-specified (deep-read; see `notes/<basename>.md`)

1. [The MASK Benchmark: Disentangling Honesty From Accuracy in AI Systems](2503.03750_mask_benchmark.pdf) — Ren et al., 2025, arXiv:2503.03750. Belief elicitation + pressure prompts; separates lying (statement ≠ belief) from inaccuracy (belief ≠ truth). Source of the belief-elicitation protocol.
2. [Liars' Bench: Evaluating Lie Detectors for Language Models](2511.16035_liars_bench.pdf) — Kretschmar, Laurito, Maiya, Marks, 2025, arXiv:2511.16035. 7 lie datasets × 4 open models; Apollo-style mean probe, follow-up probe, upper-bound probe, LLM judge. Harm-pressure sets include honest "I don't know" vs feigned ignorance.
3. [Detecting Strategic Deception Using Linear Probes](2502.03407_strategic_deception_linear_probes.pdf) — Goldowsky-Dill et al. (Apollo), 2025, arXiv:2502.03407. The standard white-box deception probe recipe (Instructed-Pairs, logistic regression, 1% FPR on Alpaca).
4. [One Probe Won't Catch Them All: Towards Targeted Deception Detection](2602.01425_one_probe_wont_catch_them_all.pdf) — Natarajan et al., 2026, arXiv:2602.01425. Probe performance depends strongly on the contrastive prompt pair; lie typology.
5. [Probing the Limits of the Lie Detector Approach to LLM Deception](2603.10003_limits_of_lie_detector_approach.pdf) — Thormann, 2026, arXiv:2603.10003. Truth probes flag lies but not true-but-misleading statements: they track falsity.
6. [Asymmetries in Spontaneous and Instructed Deception](2609.00180_spontaneous_vs_instructed_deception.pdf) — Luikham, 2026, arXiv:2609.00180. Instructed vs incentive-only deception in Llama-3.1-70B; cross-setting probe transfer.
7. [Beyond Liars' Bench: Lie Typology, Depth, and Sparsity](2607.20479_beyond_liars_bench.pdf) — Moustafa, Feser, Mai, 2026, arXiv:2607.20479. Probe/SAE variants; transfer to harm-pressure sets is below chance.
8. [Do LLMs Really Know What They Don't Know?](2510.09033_do_llms_know_what_they_dont_know.pdf) — Cheang et al., 2025, arXiv:2510.09033. Hallucination detectors track knowledge recall, not truth: good on "unassociated" hallucinations, near chance on "associated" ones.
9. ["Did you lie?" Evaluating Lie Detectors across Model Scale and Belief-Verified Model Organisms](2606.12618_did_you_lie.pdf) — Cooney, Africa, Irving (UK AISI), 2026, arXiv:2606.12618. Belief-verified TriviaQA lying protocol; self-report probe; label-noise analysis.
10. [Fine-Tuned Lie Detectors Failed to Generalize](anthropic_lie_detectors.html) — Hopkins, Khullar, Wang, Roger, Anthropic Alignment Science blog, 2026-08-21. On-policy lie datasets (released as `Noddybear/lies`); honest mistakes explicitly excluded from "lie".

## Found through search (abstract-level review only)

11. [Discovering Latent Knowledge Without Supervision (CCS)](2212.03827_ccs_discovering_latent_knowledge.pdf) — Burns et al., 2022. Unsupervised truth direction.
12. [The Internal State of an LLM Knows When It's Lying](2304.13734_internal_state_knows_when_lying.pdf) — Azaria & Mitchell, 2023. First true/false activation classifier; source of the facts used in Apollo's Instructed-Pairs.
13. [Still No Lie Detector for Language Models](2307.00175_still_no_lie_detector.pdf) — Levinstein & Herrmann, 2023. Generalisation failures of 11 and 12.
14. [How to Catch an AI Liar](2309.15840_how_to_catch_an_ai_liar.pdf) — Pacchiardi et al., 2023. Black-box follow-up-question detector; defines lie as false output despite demonstrably knowing.
15. [The Geometry of Truth](2310.06824_geometry_of_truth.pdf) — Marks & Tegmark, 2023. Linear truth directions; difference-of-means probes.
16. [AI Sandbagging](2406.07358_ai_sandbagging.pdf) — van der Weij et al., 2024. Strategic underperformance (wrong answers despite knowing).
17. [Semantic Entropy Probes](2406.15927_semantic_entropy_probes.pdf) — Kossen et al., 2024. Hallucination detector: probes predicting semantic entropy from hidden states.
18. [Truth is Universal: Robust Detection of Lies in LLMs](2407.12831_truth_is_universal.pdf) — Bürger et al., 2024. 2-D truth subspace; TTPD classifier.
19. [LLMs Know More Than They Show](2410.02707_llms_know_more_than_they_show.pdf) — Orgad et al., 2024. Exact-answer-token error probes; model may encode the right answer yet output a wrong one.
20. [Benchmarking Deception Probes via Black-to-White Performance Boosts](2507.12691_black_to_white_boosts.pdf) — Parrack et al., 2025.
21. [Caught in the Act](2508.19505_caught_in_the_act.pdf) — Boxo et al., 2025. Deception probes across 1.5B–14B models; many deception directions.
22. [Can LLMs Lie? Investigation beyond Hallucination](2509.03518_can_llms_lie_beyond_hallucination.pdf) — Huan et al., 2025. Explicitly contrasts lying with hallucination; mechanistic analysis and steering.
23. [Rift: A Conflict Signature for Deception](2606.17229_rift_conflict_signature.pdf) — 2026. Claims a residual-rank signature separating knowing lies from "naive liar" errors (fine-tuned controls, small models).
24. [Probe Generalization as Subspace Selection for OOD Deception Detection](2609.02893_probe_generalization_subspace.pdf) — Yoo & Skapars, 2026. Llama-3.1-8B-Instruct deception probes.
25. [A Lie Detector Test for Language Models: Reading Knowledge a Model Won't Reveal (PIR)](2609.21996_lie_detector_reading_withheld_knowledge.pdf) — Dingeto, 2026. Separates "will not answer" from "cannot answer" via recognition probing.
26. [Stress-Testing LLM Lie Detectors: Role-Play Failures and Spurious Correlations](2609.39807_stress_testing_lie_detectors.pdf) — von Klinski et al., 2026. Probes track confounds such as instruction compliance and response likelihood.

Paper-finder service was unavailable (HTTP 500); items 11–26 came from Semantic Scholar API searches and citation chasing.
