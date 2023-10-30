from typing import List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.layers import Reshape
from keras.layers import GlobalMaxPooling1D
import keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from src.molecule_pair import MoleculePair

class MlModel:

    def __init__(self, input_dim=64):


        self.siamese_model =self.create_siamese_model(input_dim)

    # Define the base network (shared weights)
    def create_base_network(self, input_dim, 
                                shape_global = 2):
        
        input_spectrogram = layers.Input(shape=input_dim)
        input_global_variables= layers.Input(shape=shape_global)

        target_shape = (input_dim, 1)  # Define the target shape


        x =  Dense(32, activation='relu') (input_spectrogram)
        x = Dropout(0.5) (x)
        x = Dense (32, activation='relu') (x)
        x = Dropout(0.5) (x)
        #x  = Reshape(target_shape, input_shape=(input_dim,))(input_spectrogram)
        #x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        #x = BatchNormalization() (x)
        #x = MaxPooling1D(pool_size=4)(x)
        #x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        #x = BatchNormalization() (x)
        #x = MaxPooling1D(pool_size=4)(x)
        #x = Conv1D(1, kernel_size=3, activation='relu', padding='same')(x)
        #x = Flatten() (x)
        #x = MaxPooling1D(pool_size=2)(x)
        #x= GlobalMaxPooling1D()(x)
        #x= Flatten() (x)
        #x = Dense(32, activation='relu') (x)
        #x = Dropout(0.5) (x)
        #x = Dense(32, activation='relu') (x)
        

        # embedded the features
        global_features= Dense(4, activation='relu')(input_global_variables)
        global_features = Dropout(0.5) (global_features)
        global_features= Dense(4, activation='relu')(global_features)

         #Concatenate the global variables with the base network output
        #concatenated = Concatenate()([x, global_features])

        concatenated= x

        # dense layer
        concatenated = Dense(32, activation='relu') (concatenated)
        concatenated =  Dropout(0.5) (concatenated) 
        #shared_network = keras.Sequential([
        #    layers.Dense(32, activation='relu'),
        #    layers.Dropout(0.5),
        #    layers.Dense(32, activation='relu'),
        #    layers.Dropout(0.5),
        #    layers.Dense(32, activation='relu')
        #])

        #return shared_network
        return Model([input_spectrogram,input_global_variables] , outputs=concatenated)
 
    
    # Example of using a contrastive loss
    @staticmethod
    def contrastive_loss(y_true, y_pred):
        margin = 1  # Adjust the margin according to your problem
        positive_similarity = y_pred
        negative_similarity = margin - y_pred
        return K.mean(y_true * K.square(positive_similarity) + (1 - y_true) * K.square(K.maximum(negative_similarity, 0)))

    # Define the Siamese network
    def create_siamese_model(self,input_dim, shape_global=2):
        #input_layer = layers.Input(shape=(input_dim,))
        
        # Shared embedding network for both input vectors
        #shared_network = keras.Sequential([
        #    layers.Dense(128, activation='relu'),
        #    layers.Dropout(0.5),
        #    layers.Dense(64, activation='relu'),
        #    layers.Dropout(0.5),
        #    layers.Dense(32, activation='relu')
        #])
        

        # Create the left and right inputs and apply the shared network
        left_input = layers.Input(shape=(input_dim))
        left_global = layers.Input(shape=shape_global)
        right_input = layers.Input(shape=(input_dim))
        right_global = layers.Input(shape=shape_global)

        #left_input = input_layer
        #right_input = input_layer
        # Create the twin networks (sharing the same weights)
        shared_network = self.create_base_network(input_dim, shape_global=shape_global)
        left_embedding = shared_network([left_input, left_global])
        right_embedding = shared_network([right_input, right_global])
        
        # Compute the L1 distance between the embeddings
        # distance layer based on cosine similairy
        distance = keras.layers.Dot(axes=(1, 1),
                                            normalize=True,
                                           name="cosine_similarity")([left_embedding, right_embedding])
        
        #l1_norm = lambda x: 1 - K.abs(x[0] - x[1])
        #merged = layers.Lambda(function=l1_norm, output_shape=lambda x: x[0], name='L1_distance')([left_embedding, right_embedding])
        #distance = Dense(1, activation='sigmoid', name='regression_layer')(merged)

        # Final similarity prediction
        #similarity_layer = layers.Dense(1, activation='sigmoid')
        #distance = similarity_layer(distance)
        

        model = Model(inputs=[left_input, left_global, right_input, right_global], outputs=distance)
        
        return model

    def compile(self):
        # Compile the model
        #self.siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Compile the model
        # Define a TensorBoard callback
        
        #self.siamese_model.compile(optimizer='adam', loss=MlModel.contrastive_loss, metrics=['mse'])
        self.siamese_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
        self.siamese_model.summary()


    def fit(self, molecule_pairs, epochs=10, batch_size=32, validation_split=0.2):
        #callback
        tensorboard_callback = TensorBoard(
                log_dir='./logs',  # Specify the directory for TensorBoard logs
                histogram_freq=1,  # Log histogram data every epoch
                write_graph=True,  # Write the model graph to TensorBoard+
                write_images=True,  # Save image summaries of model architecture
            )

        # Define a checkpoint callback to save the best model based on validation accuracy
        save_checkpoint = ModelCheckpoint(
            'best_model.h5',          # Filepath to save the best model
            monitor='val_mse',   # Metric to monitor (validation accuracy)
            save_best_only=True,      # Save only the best model
            mode='min',               # Maximize the monitored metric (for accuracy, use 'max')
            verbose=1                 # Verbosity level
        )

        input_pairs_left, input_global_left, input_pairs_right, input_global_right, labels = self.get_x_y_from_molecule_pairs(molecule_pairs)
     
        # Train the model
        self.siamese_model.fit(x=[input_pairs_left, 
                                  input_global_left,
                                  input_pairs_right,
                                  input_global_right,
                                  ], 
                               y=labels, 
                               epochs=epochs, batch_size=batch_size,
                               validation_split=validation_split,
                               callbacks = [tensorboard_callback, save_checkpoint])

    def predict(self, molecule_pairs):
        input_pairs_left, input_global_left, input_pairs_right, input_global_right, labels = self.get_x_y_from_molecule_pairs(molecule_pairs)
        return self.siamese_model.predict(x=[input_pairs_left, 
                                  input_global_left,
                                  input_pairs_right,
                                  input_global_right,])
    
    def load_best_model(self):
        # Load the best model
        self.siamese_model  = load_model('best_model.h5')


    def get_x_y_from_molecule_pairs(self, molecule_pairs:List[MoleculePair]):

        input_pairs_left = np.array([p.vector_0 for p in molecule_pairs])
        input_global_left = np.array([p.global_feats_0 for p in molecule_pairs])
        input_pairs_right = np.array([p.vector_1 for p in molecule_pairs])
        input_global_right = np.array([p.global_feats_1 for p in molecule_pairs])
        labels = np.array([p.similarity for p in molecule_pairs])
        return input_pairs_left, input_global_left, input_pairs_right, input_global_right, labels
