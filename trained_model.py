from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from model_architecture import build_1d_resnet
from data_utils import load_and_preprocess_data
import h5py

def train_and_save_model(X_train, Y_train, input_shape, num_classes, save_path='trained_model.h5'):
    resnet_1d_model = build_1d_resnet(input_shape, num_classes)

    optimizer = Adam(learning_rate=0.001)

    resnet_1d_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    resnet_1d_model.fit(
        X_train, Y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=8,
        callbacks=[early_stopping]
    )

    # Save the trained model using h5py
    with h5py.File(save_path, 'w') as file:
        resnet_1d_model.save(file)

    return resnet_1d_model

# Example usage:
X_train, X_test, Y_train, Y_test, label_encoder = load_and_preprocess_data()
trained_model = train_and_save_model(X_train, Y_train, input_shape=(1000, 12), num_classes=5)
