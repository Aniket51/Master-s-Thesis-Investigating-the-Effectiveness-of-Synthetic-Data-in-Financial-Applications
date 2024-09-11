import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Upload the data
# from google.colab import files
# uploaded = files.upload()

file_path = 'Book2.csv'
data = pd.read_csv(file_path)

# Select relevant columns
ts_data = data[['Close', 'Volume', 'Open', 'High', 'Low']]

# Normalize data
scaler = MinMaxScaler()
ts_data_normalized = scaler.fit_transform(ts_data)

# Convert to sequences
seq_len = 10  # Length of each sequence


def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)


sequences = create_sequences(ts_data_normalized, seq_len)

# Split into train and test sets
train_ratio = 0.8
train_size = int(len(sequences) * train_ratio)
train_data = sequences[:train_size]
test_data = sequences[train_size:]


# Define the TimeGAN model
class TimeGAN(tf.keras.Model):
    def __init__(self, seq_len, feature_dim, latent_dim):
        super(TimeGAN, self).__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Embedder network
        self.embedder = self.build_lstm((seq_len, feature_dim), latent_dim)

        # Recovery network
        self.recovery = self.build_lstm((seq_len, latent_dim), feature_dim)

        # Generator network
        self.generator = self.build_lstm((seq_len, latent_dim), latent_dim)

        # Discriminator network
        self.discriminator = self.build_lstm((seq_len, feature_dim), 1)

    def build_lstm(self, input_shape, output_dim):
        model = tf.keras.Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            LSTM(100, return_sequences=True),  # Change return_sequences to True
            Dense(output_dim)
        ])
        return model

    def call(self, x, training=None):
        h = self.embedder(x)
        x_tilde = self.recovery(h)
        z = tf.random.normal((tf.shape(x)[0], self.seq_len, self.latent_dim))
        e_hat = self.generator(z)
        y_fake = self.discriminator(e_hat)
        y_real = self.discriminator(x)
        return x_tilde, y_fake, y_real


# Instantiate TimeGAN
feature_dim = train_data.shape[-1]
latent_dim = 5
timegan = TimeGAN(seq_len, feature_dim, latent_dim)

# Loss functions
mse = tf.keras.losses.MeanSquaredError()
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optimizers
embedder_optimizer = tf.keras.optimizers.Adam()
recovery_optimizer = tf.keras.optimizers.Adam()
generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()


# Training step
@tf.function
def train_step(x):
    with tf.GradientTape(persistent=True) as tape:
        x_tilde, y_fake, y_real = timegan(x)
        # Embedder and recovery loss
        e_loss = mse(x, x_tilde)
        # Discriminator loss
        d_loss_real = bce(tf.ones_like(y_real), y_real)
        d_loss_fake = bce(tf.zeros_like(y_fake), y_fake)
        d_loss = d_loss_real + d_loss_fake
        # Generator loss
        g_loss = bce(tf.ones_like(y_fake), y_fake)
        # Total loss
        total_loss = e_loss + d_loss + g_loss

    # Calculate gradients
    embedder_grads = tape.gradient(e_loss, timegan.embedder.trainable_variables)
    recovery_grads = tape.gradient(e_loss, timegan.recovery.trainable_variables)
    generator_grads = tape.gradient(g_loss, timegan.generator.trainable_variables)
    discriminator_grads = tape.gradient(d_loss, timegan.discriminator.trainable_variables)

    # Apply gradients
    embedder_optimizer.apply_gradients(zip(embedder_grads, timegan.embedder.trainable_variables))
    recovery_optimizer.apply_gradients(zip(recovery_grads, timegan.recovery.trainable_variables))
    generator_optimizer.apply_gradients(zip(generator_grads, timegan.generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grads, timegan.discriminator.trainable_variables))

    return total_loss


# Training loop
epochs = 100  # Using fewer epochs for demonstration
batch_size = 8

for epoch in range(epochs):
    for i in range(0, len(train_data), batch_size):
        x_batch = train_data[i:i + batch_size]
        loss = train_step(x_batch)  # Train the model on each batch
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')


def generate_synthetic_data(model, num_samples):
    z = tf.random.normal((num_samples, seq_len, latent_dim))
    synthetic_data = model.generator(z)
    return synthetic_data


num_synthetic_samples = 100
synthetic_data = generate_synthetic_data(timegan, num_synthetic_samples)

# Invert scaling for synthetic data
synthetic_data = synthetic_data.numpy()
synthetic_data_unscaled = scaler.inverse_transform(synthetic_data.reshape(-1, feature_dim)).reshape(-1, seq_len,
                                                                                                    feature_dim)

# Convert synthetic data to a DataFrame and save to CSV
synthetic_data_reshaped = synthetic_data_unscaled.reshape(-1, feature_dim)
synthetic_df = pd.DataFrame(synthetic_data_reshaped, columns=['Close', 'Volume', 'Open', 'High', 'Low'])
synthetic_df.to_csv('synthetic_data.csv', index=False)
files.download('synthetic_data.csv')

plt.plot(synthetic_data_unscaled[0])
plt.title('Sine Wave')
plt.xlabel('Index')
plt.ylabel('sin(x)')
plt.grid(True)
plt.show()

print(synthetic_data_unscaled)
