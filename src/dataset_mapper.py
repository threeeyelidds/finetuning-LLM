from val_datasets import (
    load_tqa_sentences, 
    load_arc_sentences,
    load_hallu_eval,
    load_mmlu_sentences,
    load_harmful_behaviors,
    load_harmless_behaviors,
    load_tqa_mc2_sentences
)

valset_mapper = {
    "harmful_behaviors": load_harmful_behaviors,
    "harmless_behaviors": load_harmless_behaviors,
    "hallu_eal": load_hallu_eval,
    "tqa": load_tqa_sentences,
    "tqa-mc2": load_tqa_mc2_sentences,
    "arc-c": load_arc_sentences,
    "mmlu": load_mmlu_sentences
}