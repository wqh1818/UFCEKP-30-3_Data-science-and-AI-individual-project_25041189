#load csv file for model training
train_df_name = "train_split_segmented.csv"
train_df = pd.read_csv(train_df_name)
print(f"Training data loaded successfully: {train_df_name}")

val_df_name = "val_split_segmented.csv"
val_df = pd.read_csv(val_df_name)
print(f"Validation data loaded successfully: {val_df_name}")

test_df_name = "test_split_segmented.csv"
test_df = pd.read_csv(test_df_name)
print(f"Testing data loaded successfully: {test_df_name}")


print("\n For training dataset..")
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

print("\n For validation dataset..")
val_generator = val_test_datagen.flow_from_dataframe(
    val_df,
    x_col="path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\n For testing dataset..")
test_generator = val_test_datagen.flow_from_dataframe(
    test_df,
    x_col="path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\n Class indices:", train_generator.class_indices)

# Initialize base model with frozen weights
base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers:
    layer.trainable = False

# Attach custom classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Standard Callbacks
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1
)

# define function to print per epoch details
time_callback = TimeHistory()

# save the best perform model
checkpoint = ModelCheckpoint(
    "/content/drive/MyDrive/individual projects/resnet50_segmented100_best.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# start the timer to count total training time
total_start_time = time.time()

# Execute training
print("\nStarting Model Training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=24,
    callbacks=[early_stop, reduce_lr, time_callback, checkpoint]
)

# Convert training history to DataFrame
history_df = pd.DataFrame(history.history)

# Add epoch column and save to CSV
history_df['epoch'] = range(1, len(history_df) + 1)
history_df.to_csv("/content/drive/MyDrive/individual projects/resnet50_segmented100_history.csv", index=False)
print("Training history saved!")

print("Model saved successfully to Google Drive!")
print("\nTraining Complete!")

# Calculate and show total training time
total_training_time = time.time() - total_start_time
minutes = total_training_time // 60
seconds = total_training_time % 60
print(f"Total Training Time: {int(minutes)} min {seconds:.2f} sec")

# 1. Evaluate overall test accuracy and loss
print("\nEvaluating model on the Test Set...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

# 2. Plot the loss and accuracy of training and validation
# Plot Loss and Accuracy curves
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history_df['loss'], label='Train Loss')
plt.plot(history_df['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(history_df['accuracy'], label='Train Accuracy')
plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# 3. Get predictions for detailed metrics
# Reset the generator before predicting to ensure order matches
test_generator.reset()
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Print Classification Report
class_labels = list(test_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# 4. Plot Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 5. Plot normalized confusion metrix
cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
plt.figure(figsize=(8,6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Normalised Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 6. generate per class metrics and save it for further analysis
# Per-class metrics
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)
f1_per_class = f1_score(y_true, y_pred, average=None)

# Per-class accuracy (from confusion matrix)
class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Create DataFrame
per_class_df = pd.DataFrame({
    "class": class_labels,
    "accuracy": class_accuracy,
    "precision": precision_per_class,
    "recall": recall_per_class,
    "f1_score": f1_per_class
})
print("\nPer-Class Metrics:")
print(per_class_df)

# 7. Call the function to showcase the prediction with image
show_predictions(test_generator, model, class_labels)

# Save confusion metrics
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
cm_df.to_csv("/content/drive/MyDrive/individual projects/resnet50_segmented100_confusion_matrix.csv")
print("Confusion matrix saved successfully!")

# Save normalized confusion metrics
cm_norm_df = pd.DataFrame(cm_norm, index=class_labels, columns=class_labels)
cm_norm_df.to_csv("/content/drive/MyDrive/individual projects/resnet50_segmented100_cm_norm.csv")
print("Normalized confusion matrix saved successfully!")

# Save per class metrics table
per_class_df.to_csv("/content/drive/MyDrive/individual projects/resnet50_segmented100_per_class_metrics.csv", index=False)
print("Per-class metrics saved successfully!")

# Save evaluation metrics
# Calculate metrics
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Create summary row
results = {
    "model": "ResNet50_Segmented",
    "data_type": "segmented",
    "data_size": "100%",
    "test_accuracy": test_acc,
    "test_loss": test_loss,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "training_time_sec": total_training_time
}

results_df = pd.DataFrame([results])

# Save or append
results_path = "/content/drive/MyDrive/individual projects/model_comparison_results.csv"

if os.path.exists(results_path):
    existing_df = pd.read_csv(results_path)

    # Check if this model with data_size combination already exists
    mask = (existing_df["model"] == "ResNet50_Segmented") & (existing_df["data_size"] == "100%")
    if mask.any():
        # Overwrite the existing record
        print("Record found, overwriting existing record...")
        existing_df.loc[mask] = pd.DataFrame([results]).values
        existing_df.to_csv(results_path, index=False)
        print("Existing record overwritten!")
    else:
        # Combination not found, append as new row
        print("Record not found, adding new record...")
        results_df = pd.DataFrame([results])
        results_df.to_csv(results_path, mode='a', header=False, index=False)
        print("New record appended!")
else:
    # File doesn't exist, create it
    pd.DataFrame([results]).to_csv(results_path, index=False)
    print("New file created with record!")

print("Final evaluation metrics saved successfully!")
