# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from imblearn.combine import SMOTEENN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Load Dataset
# Replace 'creditcard.csv' with the correct path to your dataset
df = pd.read_csv('creditcard.csv')

# Display dataset information
print(df.head())
print(f"Dataset shape: {df.shape}")
print("Class distribution:")
print(df['Class'].value_counts())

# 2. Data Preprocessing
# Standardize the 'Amount' and 'Time' columns
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df['Time'] = scaler.fit_transform(df[['Time']])

# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# 3. Handle Imbalanced Dataset with SMOTEENN (SMOTE + Edited Nearest Neighbors)
smote_enn = SMOTEENN(sampling_strategy='auto', random_state=42)
X_res, y_res = smote_enn.fit_resample(X, y)
print(f"After SMOTEENN, class distribution:\n{pd.Series(y_res).value_counts()}")

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)



# 5. Build the LSTM Model
model = Sequential()

# First LSTM layer with Dropout
model.add(LSTM(units=256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))

# Second LSTM layer
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.3))

# Third LSTM layer
model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.3))

# Dense output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model with Adam optimizer and a lower learning rate
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# 6. Implement Early Stopping and Learning Rate Scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001)

# 7. Train the Model
history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2,
                    callbacks=[early_stopping, lr_scheduler])

# 8. Evaluate the Model
# Predict probabilities
y_pred = model.predict(X_test)

# Convert probabilities to binary classes
y_pred_class = (y_pred > 0.5).astype(int)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_report = classification_report(y_test, y_pred_class)
print("Classification Report:")
print(class_report)

# AUC-ROC Score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'AUC-ROC Score: {roc_auc}')

# Precision-Recall AUC
precision_recall_auc = average_precision_score(y_test, y_pred)
print(f'Precision-Recall AUC: {precision_recall_auc}')

# 9. Visualize Training Performance
# Plotting the accuracy over epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting the loss over epochs
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 10. Save the model for future use
model.save('optimized_credit_card_fraud_detection_model.h5')