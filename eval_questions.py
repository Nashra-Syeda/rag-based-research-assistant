EVAL_QUESTIONS = [
    {
        "question": "What dataset and BLEU score did the original Transformer achieve on WMT 2014 English-to-German translation?",
        "expected_source": "attention is all you need.pdf",
        "reference_answer": "The Transformer achieved a BLEU score of 28.4 on the WMT 2014 English-to-German translation task, establishing a new state-of-the-art, outperforming previous best results including ensembles by more than 2.0 BLEU.",
    },
    {
        "question": "What is the rank of the low-rank matrices used in LoRA, and how does it reduce trainable parameters?",
        "expected_source": "LoRA.pdf",
        "reference_answer": "LoRA introduces a rank r as a tunable hyperparameter, decomposing weight updates into two smaller matrices of rank r instead of updating the full weight matrix. Since r is much smaller than the model's hidden dimension, the number of trainable parameters is drastically reduced, scaling linearly with r instead of with the full matrix size.",
    },
    {
        "question": "Which paper introduces bidirectional pretraining with masked language modeling — BERT or GPT-3?",
        "expected_source": "BERT.pdf",
        "reference_answer": "BERT introduces bidirectional pretraining using masked language modeling, enabling pretrained deep bidirectional representations.",
    },
    {
        "question": "What training objective does ELECTRA use instead of masked language modeling, and how does this differ from BERT's approach?",
        "expected_source": ["ELECTRA.pdf", "BERT.pdf"],
        "reference_answer": "ELECTRA uses replaced token detection: a small generator replaces some tokens, and a discriminator is trained to classify every token as original or replaced. This differs from BERT's masked language modeling, where tokens are masked and the model predicts the original token only at masked positions.",
    },
    {
        "question": "Compare how RoBERTa modified BERT's pretraining procedure.",
        "expected_source": ["RoBERTa.pdf", "BERT.pdf"],
        "reference_answer": "RoBERTa is a replication study of BERT pretraining that found BERT was significantly undertrained. RoBERTa modifies BERT's procedure by using dynamic masking instead of static masking, removing the next-sentence-prediction objective, training on far more data (160GB vs 16GB), training longer, using larger batches, and using a larger byte-level BPE vocabulary.",
    },
    {
        "question": "What does the InstructGPT paper say about GPT-4's capabilities?",
        "expected_source": None,
        "reference_answer": "The InstructGPT paper does not discuss GPT-4 at all, since it predates GPT-4's release. It only discusses InstructGPT models relative to GPT-3.",
    },
    {
        "question": "According to these papers, what is the current state-of-the-art on ImageNet classification?",
        "expected_source": None,
        "reference_answer": "None of the provided papers report results on ImageNet classification; this information is not available in the corpus.",
    },
    {
        "question": "How does Sentence-BERT combine BERT with siamese network structure, and why is this useful for semantic similarity search?",
        "expected_source": "Sentence-BERT.pdf",
        "reference_answer": "Sentence-BERT fine-tunes BERT using a siamese (and triplet) network structure, where identical BERT copies sharing weights process sentences independently to produce fixed-size, comparable embeddings. This is useful because it reduces the cost of finding similar sentence pairs from about 65 hours with raw BERT to about 5 seconds with SBERT, using cosine similarity for fast comparison.",
    },
]