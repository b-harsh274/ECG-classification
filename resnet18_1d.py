#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

sampling_rate=100

# load and convert annotation data
df = pd.read_csv('ptbxl_database.csv', index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
Signals = load_raw_data(df, sampling_rate)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv('scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
df.reset_index(inplace=True)
Labels = df.scp_codes.apply(aggregate_diagnostic)


# In[ ]:


y=[]
x=[]

for i in range(len(Labels)):
    if Labels[i] == list(['CD']):
        y.append('CD')
        x.append(Signals[i,:,0])
    if Labels[i] == list(['HYP']):
        y.append('HYP')
        x.append(Signals[i,:,0])
    if Labels[i] == list(['NORM']):
        y.append('NORM')
        x.append(Signals[i,:,0])
    if Labels[i] == list(['STTC']):
        y.append('STTC')
        x.append(Signals[i,:,0])
    if Labels[i] == list(['MI']):
        y.append('MI')
        x.append(Signals[i,:,0])

x = np.array(x)
y = np.array(y)


# In[ ]:


augmented_signals = []
augmented_labels = []


for label in ['CD', 'HYP', 'MI', 'STTC']:
    target_indices = np.where(y == label)[0]
    augmentation_factor = (np.count_nonzero(y == 'NORM')  // np.count_nonzero(y == label)) -1
    
    for index in target_indices:
        repeated_signal = np.tile(x[index], (augmentation_factor, 1))
        noisy_signal = repeated_signal + 0.01 * np.random.randn(*repeated_signal.shape)
        repeated_signal_2d = repeated_signal.reshape(-1, repeated_signal.shape[-1])
        augmented_signals.append(noisy_signal)
        augmented_labels.extend([y[index]] * augmentation_factor)
    
X = np.concatenate([x] + augmented_signals)
Y = np.concatenate([y, np.array(augmented_labels)])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

Y_onehot = to_categorical(Y_encoded, num_classes=5)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=None)


# In[ ]:


from tensorflow.keras import layers, models

def build_1d_resnet18(input_shape, num_classes):
    input_tensor = layers.Input(shape=input_shape)

    # Initial Convolution
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', activation='relu')(input_tensor)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual Blocks
    x = residual_block_1d(x, 64, 1)
    x = residual_block_1d(x, 128, 2)
    x = residual_block_1d(x, 256, 2)
    x = residual_block_1d(x, 512, 2)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Fully Connected layer
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = models.Model(inputs=input_tensor, outputs=x, name='resnet18_1d')

    return model

def residual_block_1d(input_tensor, filters, strides):
    shortcut = input_tensor

    # First convolution layer
    x = layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same', activation='relu')(input_tensor)

    # Second convolution layer
    x = layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)

    # Shortcut connection if needed
    if strides != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='valid', activation='relu')(input_tensor)

    # Add shortcut to main path
    x = layers.add([x, shortcut])

    return x


# In[ ]:


input_shape = (1000, 12)
num_classes = 5  

resnet18_1d_model = build_1d_resnet18(input_shape, num_classes)

resnet18_1d_model.summary()


# In[ ]:


from tensorflow.keras.optimizers.legacy import Adam

optimizer = Adam(learning_rate=0.001)

resnet18_1d_model.compile(
    optimizer=optimizer, 
    loss='categorical_crossentropy',
    metrics=['accuracy'])


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# In[ ]:


resnet18_1d_model.fit(
    X_train, y_train,
    validation_split = 0.2,
    epochs=100,
    batch_size = 8,
    callbacks = early_stopping
)


# In[ ]:


resnet18_1d_model.evaluate(X_test, y_test)