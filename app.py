import streamlit as st

# Must be the first Streamlit command
st.set_page_config(page_title="Arabic-to-English Translator", layout="centered")

# Now import other libraries
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from tensorflow.keras import layers
import numpy as np

def compute_scaled_attention(q_vectors, k_vectors, v_vectors, attention_mask):
    # Compute the dot product of queries and keys
    score_matrix = tf.matmul(q_vectors, k_vectors, transpose_b=True)

    # Determine scaling factor
    scaling_factor = tf.cast(tf.shape(k_vectors)[-1], tf.float32)

    # Scale the score matrix
    scaled_scores = score_matrix / tf.math.sqrt(scaling_factor)

    # Apply masking if provided
    if attention_mask is not None:
        scaled_scores += (attention_mask * -1e9)

    # Compute attention-weighted values
    attention_output = tf.matmul(tf.nn.softmax(scaled_scores, axis=-1), v_vectors)

    return attention_output


class MultiHeadSelfAttention(layers.Layer):

    def __init__(self, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads

    def build(self, input_dims):
        self.model_dim = input_dims[-1]
        assert self.model_dim % self.num_heads == 0, "Model dimension must be divisible by number of heads."

        # Calculate head dimension
        self.head_dim = self.model_dim // self.num_heads

        # Linear transformations for Q, K, V
        self.query_transform = layers.Dense(units=self.model_dim)
        self.key_transform = layers.Dense(units=self.model_dim)
        self.value_transform = layers.Dense(units=self.model_dim)

        # Final linear transformation after multi-head attention
        self.output_transform = layers.Dense(units=self.model_dim)

    def split_heads(self, tensor_input, batch_size):
        reshaped_tensor = tf.reshape(tensor_input, shape=(batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(reshaped_tensor, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_length, head_dim)

    def call(self, q_input, k_input, v_input, attention_mask):
        batch_size = tf.shape(q_input)[0]

        # Compute Q, K, and V matrices
        q_matrix = self.query_transform(q_input)
        k_matrix = self.key_transform(k_input)
        v_matrix = self.value_transform(v_input)

        # Split into multiple heads
        q_matrix = self.split_heads(q_matrix, batch_size)
        k_matrix = self.split_heads(k_matrix, batch_size)
        v_matrix = self.split_heads(v_matrix, batch_size)

        # Compute attention using scaled dot-product function
        attention_scores = compute_scaled_attention(q_matrix, k_matrix, v_matrix, attention_mask)

        # Rearrange dimensions
        attention_scores = tf.transpose(attention_scores, perm=[0, 2, 1, 3])

        # Concatenate multiple heads
        concatenated_attention = tf.reshape(attention_scores, shape=(batch_size, -1, self.model_dim))

        # Final transformation
        final_output = self.output_transform(concatenated_attention)

        return final_output


class PositionalEncoding(layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

    def compute_angles(self, position, index, model_dim):
        # Compute the angle rates for different dimensions
        angle_rates = 1 / np.power(10000., (2 * (index // 2)) / np.float32(model_dim))
        return position * angle_rates  # Shape: (sequence_length, model_dim)

    def call(self, inputs):
        # Extract sequence length and model dimension
        sequence_length = inputs.shape[-2]
        model_dim = inputs.shape[-1]

        # Compute the angles using broadcasting
        angle_matrix = self.compute_angles(
            np.arange(sequence_length)[:, np.newaxis],  # Position indices
            np.arange(model_dim)[np.newaxis, :],       # Dimension indices
            model_dim
        )

        # Apply sine to even indices and cosine to odd indices
        angle_matrix[:, 0::2] = np.sin(angle_matrix[:, 0::2])
        angle_matrix[:, 1::2] = np.cos(angle_matrix[:, 1::2])

        # Expand for batch compatibility
        pos_encoding = angle_matrix[np.newaxis, ...]

        return inputs + tf.cast(pos_encoding, dtype=tf.float32)


class EncoderLayer(layers.Layer):
    def __init__(self, ffn_units, num_heads, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.ffn_units = ffn_units  # Feed Forward Network units
        self.num_heads = num_heads  # Number of attention heads
        self.dropout_rate = dropout_rate  # Dropout rate

        # Multi-head Attention layer
        self.multi_head_attention = MultiHeadSelfAttention(self.num_heads)
        self.dropout_1 = layers.Dropout(rate=self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)  # Layer Norm 1

        # Feed Forward Network (FFN)
        self.ffn_dense1 = layers.Dense(units=self.ffn_units, activation="relu")
        self.ffn_dense2 = layers.Dense(units=None)  # Output size is determined in `build`
        self.dropout_2 = layers.Dropout(rate=self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)  # Layer Norm 2

    def build(self, input_shape):
        self.d_model = input_shape[-1]  # Get embedding dimension dynamically
        self.ffn_dense2.units = self.d_model  # Set FFN output units dynamically

    def call(self, inputs, mask=None, training=False):
        # Multi-head self-attention
        attention_output = self.multi_head_attention(inputs, inputs, inputs, mask)
        attention_output = self.dropout_1(attention_output, training=training)
        attention_output = self.norm_1(inputs + attention_output)  # Add & Norm

        # Feed Forward Network
        ffn_output = self.ffn_dense1(attention_output)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout_2(ffn_output, training=training)
        ffn_output = self.norm_2(attention_output + ffn_output)  # Add & Norm

        return ffn_output


class Encoder(layers.Layer):
    def __init__(self, n_layers, ffn_units, num_heads, dropout_rate, vocab_size, d_model, name="encoder"):
        super(Encoder, self).__init__(name=name)
        self.n_layers = n_layers
        self.d_model = d_model

        # Embedding and Positional Encoding
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)

        # Stacking multiple Encoder Layers
        self.enc_layers = [
            EncoderLayer(ffn_units, num_heads, dropout_rate) for _ in range(n_layers)
        ]

    def call(self, inputs, mask=None, training=False):
        # Convert tokens into embeddings
        embeddings = self.embedding(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # Scale embeddings
        embeddings = self.pos_encoding(embeddings)  # Add positional encodings
        embeddings = self.dropout(embeddings, training=training)

        # Pass through stacked encoder layers
        output = embeddings
        for layer in self.enc_layers:
            output = layer(output, mask, training=training)

        return output

class DecoderLayer(layers.Layer):
    def __init__(self, ffn_units, num_heads, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.ffn_units = ffn_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.d_model = input_shape[-1]

        # Self-attention (masked causal attention)
        self.self_attention = MultiHeadSelfAttention(self.num_heads)
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

        # Encoder-decoder attention
        self.enc_dec_attention = MultiHeadSelfAttention(self.num_heads)
        self.dropout_2 = layers.Dropout(self.dropout_rate)
        self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

        # Feedforward network
        self.ffn1 = layers.Dense(units=self.ffn_units, activation="relu")
        self.ffn2 = layers.Dense(units=self.d_model)
        self.dropout_3 = layers.Dropout(self.dropout_rate)
        self.norm_3 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, enc_outputs, mask_1=None, mask_2=None, training=False):
        # Masked self-attention (causal attention)
        attn_output = self.self_attention(inputs, inputs, inputs, mask_1)
        attn_output = self.dropout_1(attn_output, training=training)
        attn_output = self.norm_1(attn_output + inputs)  # Residual connection

        # Encoder-decoder attention
        attn_output_2 = self.enc_dec_attention(attn_output, enc_outputs, enc_outputs, mask_2)
        attn_output_2 = self.dropout_2(attn_output_2, training=training)
        attn_output_2 = self.norm_2(attn_output_2 + attn_output)  # Residual connection

        # Feedforward network
        ffn_output = self.ffn1(attn_output_2)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout_3(ffn_output, training=training)
        output = self.norm_3(ffn_output + attn_output_2)  # Residual connection

        return output

class Decoder(layers.Layer):
    def __init__(self,
                 n_layers,
                 ffn_units,
                 num_heads,
                 dropout_rate,
                 vocab_size,
                 d_model,
                 name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.d_model = d_model
        self.n_layers = n_layers

        # Embedding layer
        self.embedding = layers.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding()
        self.dropout = layers.Dropout(rate=dropout_rate)

        # Stacked Decoder layers
        self.dec_layers = [DecoderLayer(ffn_units, num_heads, dropout_rate)
                           for _ in range(n_layers)]

    def call(self, inputs, enc_outputs, mask_1=None, mask_2=None, training=False):
        # Compute embeddings and scale
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # Apply positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        # Pass through stacked decoder layers
        for layer in self.dec_layers:
            x = layer(x, enc_outputs, mask_1, mask_2, training=training)

        return x

class Transformer(tf.keras.Model):
    def __init__(self,
                 vocab_size_enc,
                 vocab_size_dec,
                 d_model,
                 n_layers,
                 ffn_units,
                 num_heads,
                 dropout_rate,
                 name="transformer"):
        super(Transformer, self).__init__(name=name)

        # Encoder
        self.encoder = Encoder(n_layers, ffn_units, num_heads, dropout_rate, vocab_size_enc, d_model)

        # Decoder
        self.decoder = Decoder(n_layers, ffn_units, num_heads, dropout_rate, vocab_size_dec, d_model)

        # Final linear layer
        self.final_layer = layers.Dense(units=vocab_size_dec, name="output_layer")

    def create_padding_mask(self, seq):
        """Create a mask for padding tokens (0s)."""
        return tf.cast(tf.math.equal(seq, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, seq):
        """Create a look-ahead mask for causal attention."""
        seq_len = tf.shape(seq)[1]
        return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    def call(self, enc_inputs, dec_inputs, training=False):
        """Forward pass through the Transformer model."""
        # Create padding masks
        enc_padding_mask = self.create_padding_mask(enc_inputs)
        dec_padding_mask = self.create_padding_mask(enc_inputs)  # Encoder-decoder attention

        # Create look-ahead mask
        look_ahead_mask = self.create_look_ahead_mask(dec_inputs)
        combined_mask = tf.maximum(self.create_padding_mask(dec_inputs), look_ahead_mask)

        # Pass through encoder
        enc_outputs = self.encoder(enc_inputs, enc_padding_mask, training=training)

        # Pass through decoder
        dec_outputs = self.decoder(dec_inputs, enc_outputs, combined_mask, dec_padding_mask, training=training)

        # Final output projection
        return self.final_layer(dec_outputs)




# Load Tokenizers
def load_tokenizer(filename):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
        os.path.join(current_dir, filename)
    )
    return tokenizer

# Set hyperparamters for the model
D_MODEL = 512 # 512
N_LAYERS = 4 # 6
FFN_UNITS = 512 # 2048
N_HEADS = 8 # 8
DROPOUT_RATE = 0.1 # 0.1
MAX_LENGTH = 15  # Maximum sequence length

# Load Model
@st.cache_resource
def load_model(source_vocab_size, target_vocab_size):
    transformer = Transformer(
        vocab_size_enc=source_vocab_size + 2,
        vocab_size_dec=target_vocab_size + 2,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        ffn_units=FFN_UNITS,
        num_heads=N_HEADS,
        dropout_rate=DROPOUT_RATE
    )

    # Dummy input to build the model
    dummy_enc_input = tf.ones((1, MAX_LENGTH), dtype=tf.int32)
    dummy_dec_input = tf.ones((1, MAX_LENGTH), dtype=tf.int32)
    transformer(dummy_enc_input, dummy_dec_input, training=False)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "model_files", "arabic_to_english_transformer_weights.weights.h5")
    # Load weights
    transformer.load_weights(model_path)
    return transformer

# Load tokenizers
source_tokenizer_path = os.path.join("model_files", "source_tokenizer.subword")
target_tokenizer_path = os.path.join("model_files", "target_tokenizer.subword")

source_tokenizer = load_tokenizer(source_tokenizer_path)
target_tokenizer = load_tokenizer(target_tokenizer_path)
model = load_model(source_tokenizer.vocab_size, target_tokenizer.vocab_size)

num_words_inputs = source_tokenizer.vocab_size + 2
num_words_output = target_tokenizer.vocab_size + 2

start_token_source = [num_words_inputs - 2]
end_token_source = [num_words_inputs - 1]

start_token_target = [num_words_output - 2]
end_token_target = [num_words_output - 1]

def predict(inp_sentence, tokenizer_in, tokenizer_out, target_max_len):
    # Tokenize and add special tokens
    inp_sentence = start_token_source + tokenizer_in.encode(inp_sentence) + end_token_source
    enc_input = tf.expand_dims(inp_sentence, axis=0)

    # Initialize decoder input with SOS token
    out_sentence = start_token_target
    output = tf.expand_dims(out_sentence, axis=0)

    for _ in range(target_max_len):
        # Get predictions from Transformer
        predictions = model(enc_input, output, training=False)

        # Get the highest probability token
        prediction = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

        # Check for EOS token and break
        if predicted_id == end_token_target:
            return tf.squeeze(output, axis=0)

        # Append predicted token to output
        output = tf.concat([output, predicted_id], axis=-1)

    # Decode token IDs to text
    return tf.squeeze(output, axis=0)


def translate(sentence):
    # Get the predicted sequence for the input sentence
    output = predict(sentence, source_tokenizer, target_tokenizer,MAX_LENGTH)

    # Convert token IDs back to text, stopping at the EOS token
    predicted_sentence = target_tokenizer.decode(
        [i for i in output if i < start_token_target]  # Stop at EOS token
    )

    return predicted_sentence

# Applying custom CSS styles
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    .stApp {
        background-color: black;
    }
    .stTextArea textarea {
        background-color: #444444;
        color: white;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #ff6600;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #cc5500;
    }
    .stSuccess {
        background-color: #90EE90;
        color: black;
        padding: 10px;
        border-radius: 10px;
        font-weight: bold;
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌍 Arabic to English Translator")
st.write("Enter Arabic text below and get an English translation.")

# Text input
user_input = st.text_area("📝 Input Arabic Text", height=100)

if st.button("Translate 🚀"):
    if user_input.strip():
        translation = translate(user_input)
        st.success(f"**Translated Text:**\n\n{translation}")
    else:
        st.warning("⚠️ Please enter some text before translating.")