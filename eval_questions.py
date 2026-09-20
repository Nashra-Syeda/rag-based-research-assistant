EVAL_QUESTIONS = [
    {
        "question": "What dataset and BLEU score did the original Transformer achieve on WMT 2014 English-to-German translation?",
        "expected_source": "attention is all you need.pdf",
    },
    {
        "question": "What is the rank of the low-rank matrices used in LoRA, and how does it reduce trainable parameters?",
        "expected_source": "LoRA.pdf",
    },
    {
        "question": "Which paper introduces bidirectional pretraining with masked language modeling — BERT or GPT-3?",
        "expected_source": "BERT.pdf",
    },
    {
        "question": "What training objective does ELECTRA use instead of masked language modeling, and how does this differ from BERT's approach?",
        "expected_source": ["ELECTRA.pdf", "BERT.pdf"],
    },
    {
        "question": "Compare how RoBERTa modified BERT's pretraining procedure.",
        "expected_source": ["RoBERTa.pdf", "BERT.pdf"],
    },
    {
        "question": "What does the InstructGPT paper say about GPT-4's capabilities?",
        "expected_source": None,
    },
    {
        "question": "According to these papers, what is the current state-of-the-art on ImageNet classification?",
        "expected_source": None,
    },
    {
        "question": "How does Sentence-BERT combine BERT with siamese network structure, and why is this useful for semantic similarity search?",
        "expected_source": "Sentence-BERT.pdf",
    },
]