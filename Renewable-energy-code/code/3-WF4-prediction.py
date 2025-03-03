import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Input, Concatenate
from keras.models import Model
from deap import algorithms, base, creator, tools
import random
# 1. Load the data
data = pd.read_csv('../Experimental data/WF4-66_filled_pre.csv')
weather_data = data[['Wind speed - at the height of wheel hub  (m/s)']]
merged_imfs = pd.read_csv('../Result-file/WF4-Merged_IMFs.csv')
power_data = data[['Power (MW)']]
target_data = data[['target']]

print(f"weather-length: {len(weather_data)}")
print(f"merged-length: {len(merged_imfs)}")
print(f"power-length: {len(power_data)}")
print(f"target-length: {len(target_data)}")

input_data = pd.concat([weather_data, merged_imfs, power_data], axis=1)
real_values = target_data['target'].values
train_size = int(0.6 * len(input_data))
val_size = int(0.2 * len(input_data))
X_train = input_data.iloc[:train_size]
y_train = real_values[:train_size]
X_val = input_data.iloc[train_size:train_size + val_size]
y_val = real_values[train_size:train_size + val_size]
X_test = input_data.iloc[train_size + val_size:]
y_test = real_values[train_size + val_size:]
y_train_df = pd.DataFrame(y_train, columns=['target'])
y_test_df = pd.DataFrame(y_test, columns=['target'])
y_train_df.to_csv('../Experimental data/y_train.csv', index=False)
y_test_df.to_csv('../Experimental data/y_test.csv', index=False)
# 3. Defining the GAN Model
def build_gan(input_shape, output_shape):
    def build_generator():
        input_layer = Input(shape=(input_shape,))
        x = Dense(256, activation='relu')(input_layer)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(output_shape, activation='linear')(x)
        generator = Model(input_layer, output_layer)
        return generator

    def build_discriminator():
        input_layer = Input(shape=(output_shape,))
        x = Dense(128, activation='relu')(input_layer)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        output_layer = Dense(1, activation='sigmoid')(x)
        discriminator = Model(input_layer, output_layer)
        return discriminator
    # Compile GAN
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    discriminator.trainable = False
    gan_input = Input(shape=(input_shape,))
    gan_output = discriminator(generator(gan_input))
    gan = Model(gan_input, gan_output)

    discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error',
                          metrics=['accuracy'])
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error')
    return generator, discriminator, gan
# 4. Training
def train_gan(generator, discriminator, gan, X_train, y_train, X_val, y_val, epochs=1000, batch_size=128):
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_samples = X_train.iloc[idx, -merged_imfs.shape[1]:].values
        noise = np.random.normal(0, 1, (batch_size, X_train.shape[1] - merged_imfs.shape[1]))
        fake_samples = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        noise = np.random.normal(0, 1, (batch_size, X_train.shape[1] - merged_imfs.shape[1]))
        g_loss = gan.train_on_batch(noise, real_labels)
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
# 5. Define the NSGA-II optimization
def nsga2_optimization(predicted_imfs, real_values, n_generations=100, pop_size=200):
    # Define the objective function
    def evaluate(individual):
        combined_signal = np.sum(predicted_imfs * individual, axis=1)
        error = np.mean((combined_signal - real_values) ** 2)
        weight_sum = np.sum(individual)
        return error, weight_sum
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=predicted_imfs.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=20.0, indpb=1.0 / predicted_imfs.shape[1])
    toolbox.register("select", tools.selNSGA2)
    population = toolbox.population(n=pop_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.9, mutpb=0.1,
                              ngen=n_generations, verbose=False)
    best_individual = tools.selBest(population, k=1)[0]
    return best_individual

if __name__ == "__main__":
    # Building GAN Model
    input_shape = X_train.shape[1] - merged_imfs.shape[1]
    output_shape = merged_imfs.shape[1]
    generator, discriminator, gan = build_gan(input_shape, output_shape)
    # Training GAN
    train_gan(generator, discriminator, gan, X_train, y_train, X_val, y_val)
    # Using GAN to predict signal components
    noise = np.random.normal(0, 1, (X_test.shape[0], input_shape))
    predicted_imfs = generator.predict(noise)
    # The combination was optimized using NSGA-II
    best_weights = nsga2_optimization(predicted_imfs, y_test)
    # Calculate the final prediction
    final_prediction = np.sum(predicted_imfs * best_weights, axis=1)
    print("MAE：" + str(np.mean((np.abs(y_test - final_prediction)) / 66)))
    print("MSE：" + str(np.mean(((y_test - final_prediction) / 66) ** 2)))
    print("RMSE：" + str(np.sqrt(np.mean(((y_test - final_prediction) / 66) ** 2))))
    result_df = pd.DataFrame({"Real_Value": y_test, "Predicted_Value": final_prediction})
    result_df.to_csv("../Result-file/Result.csv", index=False)


