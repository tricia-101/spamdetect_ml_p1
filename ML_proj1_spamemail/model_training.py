import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore

# Load values from the previous step
data = np.load("processed_data.npz")
train_sequences = data['train_X']
test_sequences = data['test_X']
train_Y = data['train_Y']
test_Y = data['test_Y']
vocab_size = int(data['vocab_size'][0])
max_len = 100


## Defining the Model

#a sequential model is used meaning that the data is input passes through each stage before a prediction is given
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len),
    tf.keras.layers.LSTM(16),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer
])
#the LSTM ensures the program keeps the memory of the last 16 words to understand things in context
#the dense layer takes the patterns established by the LSTM and forms deep reasoning with them, the activation just enforces that the relationship isn't linear
#the second dense layer or output layer squeezes the final output number between 1 and 0

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), #Scoring system
    optimizer='adam', #adam is an algorithm that takes loss scores and calculates how to tweak them according to internal weights to improve accuracy
    metrics=['accuracy'] #prints overall percentage of accuracy to track performance
)

model.summary()


## Training the Model

es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True) #if the code fails to improve after 3 cycles it stops it
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=0) #if the error rate stops dropping the learning rate drops to make more precise measurements

history = model.fit(
    train_sequences, train_Y,
    validation_data=(test_sequences, test_Y),
    epochs=20, #reads through data max 20 times
    batch_size=32, #model reads in batches of 32
    callbacks=[lr, es]
)


## Evaluate the model

test_loss, test_accuracy = model.evaluate(test_sequences, test_Y)
print('Test Loss :',test_loss) #final test loss percentages
print('Test Accuracy :',test_accuracy) #final test accuracy percentage

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

#graphs it in a graph