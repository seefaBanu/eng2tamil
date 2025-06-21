import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Add CORS import
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

app = Flask(__name__)
CORS(app, resources={r"/translate": {"origins": ["http://localhost:3000", "https://your-react-app-url"]}})  # Enable CORS for specific origins

# Model Parameters
max_vocab_size = 10000
MAX_LENGTH = 30
NUM_LAYERS = 4
D_MODEL = 128
NUM_HEADS = 8
DFF = 512
PE_INPUT = 1000
PE_TARGET = 1000
DROPOUT_RATE = 0.3

# Load tokenizers
checkpoint_path = "transformer_checkpoints"
try:
    with open(os.path.join(checkpoint_path, 'en_tokenizer.pkl'), 'rb') as f:
        en_tokenizer = pickle.load(f)
    with open(os.path.join(checkpoint_path, 'ta_tokenizer.pkl'), 'rb') as f:
        ta_tokenizer = pickle.load(f)
except FileNotFoundError:
    print("Tokenizer files not found. Fitting new tokenizers using train2.csv.")
    if not os.path.exists('train2.csv'):
        raise FileNotFoundError("train2.csv not found for tokenizer fitting.")
    data = pd.read_csv('train2.csv')
    train_english = data['en'].astype(str).to_list()
    train_tamil = [f"<start> {sentence} <end>" for sentence in data['ta'].str.strip().to_list()]
    en_tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')
    ta_tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')
    en_tokenizer.fit_on_texts(train_english)
    ta_tokenizer.fit_on_texts(train_tamil)
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(os.path.join(checkpoint_path, 'en_tokenizer.pkl'), 'wb') as f:
        pickle.dump(en_tokenizer, f)
    with open(os.path.join(checkpoint_path, 'ta_tokenizer.pkl'), 'wb') as f:
        pickle.dump(ta_tokenizer, f)

INPUT_VOCAB_SIZE = min(max_vocab_size, len(en_tokenizer.word_index) + 1)
TARGET_VOCAB_SIZE = min(max_vocab_size, len(ta_tokenizer.word_index) + 1)
print(f"English vocab size: {INPUT_VOCAB_SIZE}")
print(f"Tamil vocab size: {TARGET_VOCAB_SIZE}")

# Positional Encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

# Multi-Head Attention Layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

# Feed-Forward Network
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

# Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.3):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Decoder Layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.3):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2

# Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.3):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        return x

# Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.3):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training=training,
                                                  look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        return x, attention_weights

# Transformer Model
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.3):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training=training, mask=enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training=training,
                                                    look_ahead_mask=look_ahead_mask, padding_mask=dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights

# Masks
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

# Initialize and build the Transformer model
transformer = Transformer(
    num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF,
    input_vocab_size=INPUT_VOCAB_SIZE, target_vocab_size=TARGET_VOCAB_SIZE,
    pe_input=PE_INPUT, pe_target=PE_TARGET, rate=DROPOUT_RATE
)

# Build the model
sample_input = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
sample_target = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
enc_padding_mask, combined_mask, dec_padding_mask = create_masks(sample_input, sample_target)
transformer(sample_input, sample_target, training=False,
            enc_padding_mask=enc_padding_mask, look_ahead_mask=combined_mask, dec_padding_mask=dec_padding_mask)
print("Model built successfully.")

# Load the latest weights
weights_path = os.path.join(checkpoint_path, 'transformer_epoch_140.weights.h5')

try:
    transformer.load_weights(weights_path)
    print(f"Loaded weights from {weights_path}")
except Exception as e:
    print(f"Error loading weights: {e}")
    raise

# Beam Search Evaluation
def beam_search_evaluate(sentence, beam_width=3):
    try:
        sentence = en_tokenizer.texts_to_sequences([sentence])[0]
        if not sentence:
            return "<empty input>"
        sentence = pad_sequences([sentence], maxlen=MAX_LENGTH, padding='post')
        encoder_input = tf.convert_to_tensor(sentence)
        start_token = [ta_tokenizer.word_index.get('<start>', 1)]
        beams = [(tf.expand_dims(start_token, 0), 0.0)]  # (sequence, score)
        for _ in range(MAX_LENGTH):
            new_beams = []
            for output, score in beams:
                enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
                predictions, _ = transformer(
                    encoder_input, output, training=False,
                    enc_padding_mask=enc_padding_mask, look_ahead_mask=combined_mask, dec_padding_mask=dec_padding_mask)
                predictions = predictions[:, -1:, :]
                top_k_probs, top_k_ids = tf.nn.top_k(tf.nn.softmax(predictions, axis=-1), k=beam_width)
                for i in range(beam_width):
                    new_output = tf.concat([output, top_k_ids[:, :, i]], axis=-1)
                    new_score = score + tf.math.log(top_k_probs[:, :, i]).numpy()[0][0]
                    new_beams.append((new_output, new_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if beams[0][0].numpy()[-1][-1] == ta_tokenizer.word_index.get('<end>', 2):
                break
        best_sequence = beams[0][0].numpy()[0][1:]  # Skip <start>
        predicted_sentence = ta_tokenizer.sequences_to_texts([best_sequence])[0]
        return predicted_sentence if predicted_sentence else "<no output>"
    except Exception as e:
        print(f"Evaluation error: {e}")
        return "<error>"

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    sentence = data.get('sentence', '')
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    translation = beam_search_evaluate(sentence)
    return jsonify({'input': sentence, 'translation': translation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)