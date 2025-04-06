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




I want to create a new architecture and training pipeline based on Segment Anything and SegGPT.

Both models are segmenters that use ViT and can use MAE. My pipeline will be a binary classifier

The inspiration models are segmenters, however, my new architecture will be a promptable binary classifier, not specialized, that is, a binary classifier that can classify images out of distribution trained only with positive and negative examples.

-Backbone vit
We will have a common vit backbone for image query and for positive and negative prompts. We will not use the cls token as output, we will use all embeddings.

-PromptEncoder
Promptable encoder will have the function of learning and calculating similarity matrix between positive and negative prompts (pp, pn), and calculating statistics for positive and negative prompts that will be used for comparison with image query.

For all operations in PromptEncoder, the positive and negative embeds will be normalized, and then a similarity matrix will be calculated between the embeds of the positive and negative prompts, in order to map possible thresholds for the binary classifier. After a projection of the pp and pn, the statistics of each feature of the embeddings mean, std... will be extracted. At the end, the output of the prompt encoder will be the statistics and the thershold similarity matrix.

When exiting the promptencoder, the embeds of the image query will be concatenated with the statistics of the pp and pn, and then the number of tokens will increase. The output of the transformers will be concaternated with the projection of the threshold matrix and will be fed into a binary classifier that will classify 0 or 1. (1 if the query image is from the same class as the positive prompts and 0 if the query is different).

Class names:

PromptableBinaryClassifier
ViTBackbone
PromptEncoder
ClassifierHead

PromptDataset is already implemented

train_dataset = PromptDataset("datasets/final/terumo/train", num_prompts=2)
    BATCH_SIZE = 12
    num_prompts=2
    model.to('cuda')
    for batch in train_loader:
        
        query_img = batch['query'].to('cuda')
        pos_prompts = batch['pos_imgs'].to('cuda')
        neg_prompts = batch['neg_imgs'].to('cuda')
        labels = batch['query_label'].to('cuda')
        
        print("\nPrompt shapes:")
        print(f"Query: {query_img.shape}")
        print(f"Positives: {pos_prompts.shape}")
        print(f"Negatives: {neg_prompts.shape}")
        logits = model(query_img, pos_prompts, neg_prompts)
        print("Logits:", logits.shape)
        print("Labels:", labels.shape)



For the code above, the result with data format is as follows:
Prompt shapes:
Query: torch.Size([12, 3, 224, 224])
Positives: torch.Size([12, 2, 3, 224, 224])
Negatives: torch.Size([12, 2, 3, 224, 224])
Logits: torch.Size([12, 1])
Labels: torch.Size([12])
