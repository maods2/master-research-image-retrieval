┌──────────────────────────────────────────────────────────────────────────────┐
│                        Promptable Binary Classifier                          │
├──────────────────────────────┬──────────────────────────────┬────────────────┤
│          Query Image         │      Positive Prompts        │ Negative Prompts
│          [3, 224, 224]       │    [N, 3, 224, 224]          │ [N, 3, 224, 224]
└───────────────┬──────────────┴──────────────┬───────────────┴───────┬────────┘
                │                             │                       │
          ┌─────▼─────┐                 ┌─────▼─────┐           ┌─────▼─────┐
          │ ViT       │                 │ ViT       │           │ ViT       │
          │ Backbone  │                 │ Backbone  │           │ Backbone  │
          └─────┬─────┘                 └─────┬─────┘           └─────┬─────┘
                │                             │                       │
          ┌─────▼─────┐                 ┌─────▼─────┐           ┌─────▼─────┐
          │ Query     │                 │ Positive  │           │ Negative  │
          │ Embedding │                 │ Embeddings│           │ Embeddings│
          └─────┬─────┘                 └─────┬─────┘           └─────┬─────┘
                │                             └──────────┐    ┌───────┘
                │                                        │    │
                │                              ┌─────────▼────▼────────┐
                │                              │     Prompt Encoder    │
                │                              ├───────────────────────┤
                │                              │  - Cross-attention    │
                │                              │  - Cosine Similarity  │
                │                              │  - Transformer Layers │
                │                              │  - Statistical Pooling│
                │                              └──────────┬────────────┘
                │                                         │
                │                               ┌──────────▼────────┐
                │                               │ Prompt Context    │
                │                               │   Embedding       │
                │                               └──────────┬────────┘
                │                                         │
                └───────────────────┐           ┌─────────▼────────┐
                                    │           │ Attention Fusion │
                                    ├───────────►  (Multi-head)    │
                                                └─────────┬────────┘
                                                          │
                                                    ┌─────▼─────┐
                                                    │ Classifier│
                                                    │  Head     │
                                                    └─────┬─────┘
                                                          │
                                                    ┌─────▼─────┐
                                                    │ Binary    │
                                                    │ Output    │
                                                    └───────────┘



┌──────────────────────────────┐
│        Prompt Encoder        │
├──────────────────────────────┤
│ 1. Project Pos/Neg Embeddings│
│ 2. Cross-Attention Mechanism │      ┌───────────────┐
│ 3. Cosine Similarity Matrix  ├─────►│ Similarity-   │
│ 4. Context Masking           │      │ Weighted Context
│ 5. Feature Concatenation     │      └───────────────┘
│ 6. Transformer Processing    │
│ 7. Statistical Aggregation   │
│   (Mean+Max+Std pooling)     │
└──────────────────────────────┘



