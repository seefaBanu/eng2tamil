# eng2tamil

This project develops a neural machine translation (NMT) system to translate English sentences into Tamil, a low-resource Dravidian language. The model leverages a Transformer architecture, enhanced with data augmentation and advanced decoding techniques, to achieve robust performance. The project addresses challenges in low-resource translation by incorporating back-translation and a scalable model design.

## live on : https://eng2tam.netlify.app/

<img width="680" height="243" alt="Screenshot 2026-01-18 at 8 54 54 AM" src="https://github.com/user-attachments/assets/75716753-9ea8-44c3-8cfe-c46a8add9177" />

## Project Architecture

- Transformer: A custom Transformer with 4 layers, 8 attention heads, 128-dimensional embeddings, and a feed-forward dimension of 512. It includes:
    Positional encoding to capture word order.
    Multi-head self-attention and cross-attention mechanisms.
    Dropout (rate = 0.3) for regularization.
- Masks: Padding and look-ahead masks ensure proper attention during training and inference.

  <img width="1060" height="451" alt="Screenshot 2026-01-18 at 9 00 23 AM" src="https://github.com/user-attachments/assets/4c353b29-07c4-40e3-9bf6-1793b326486e" />


<img width="845" height="307" alt="Screenshot 2026-01-18 at 8 55 14 AM" src="https://github.com/user-attachments/assets/1e4bb5b2-6d94-48da-8f9b-f092f7506943" />

## Methodology

### Data Augmentation
- Back-Translation: The M2M100_418M model translates Tamil sentences to synthetic English, doubling the dataset size. This mitigates data scarcity and enhances model robustness.
- Fallback: If back-translation fails, the original dataset is used.

### Tokenization
- Tokenizer: Keras Tokenizer with a vocabulary size of 10,000 for both languages, handling out-of-vocabulary (OOV) tokens.
- Encoding: Sentences are converted to sequences, padded to a maximum length of 30 tokens, and clipped to the vocabulary size.
