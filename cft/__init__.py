"""construct-from-text (cft)
===========================

Reference implementation for *Measuring Latent Constructs from Text: A
Multi-Corpus Benchmark of Dictionary, Topic Model, Fine-Tuned Encoder and
LLM-Based Measurement with Validity Evidence*.

The package packs one workflow:

    codebook  ->  four measurement arms  ->  reliability  ->  validity  ->
    error propagation  ->  reporting

so that a study which extracts a theoretical construct from text can report the
same evidence a study using a questionnaire is expected to report.

Modules
-------
``codebook``     construct definitions, coding rules and the expert lexicon
``corpora``      one schema for every corpus, with the sampling/split logic
``dictionary``   arm 1  -- negation-aware expert dictionary
``topic``        arm 2  -- topic model with anchor-based construct mapping
``encoder``      arm 3  -- fine-tuned transformer with masked regression heads
``llm``          arm 4  -- LLM rater, prompt variants, perturbation stimuli
``reliability``  Krippendorff alpha, run stability, prompt sensitivity
``validity``     MTMM, Campbell-Fiske rules, HTMT, criterion validity, invariance
``simulate``     Monte Carlo attenuation, method-factor bias, AUC loss
``report``       tidy long-format helpers and the reporting checklist
``viz``          the figure design system
"""

__version__ = "0.1.0"

from . import (codebook, corpora, dictionary, pipeline, reliability, report,  # noqa: F401
               simulate, topic, validity, viz)

__all__ = ["codebook", "corpora", "dictionary", "topic", "reliability",
           "validity", "simulate", "report", "pipeline", "viz", "__version__"]
