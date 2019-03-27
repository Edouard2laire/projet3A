import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np


def get_encoder(sentence_lenght=20,input_dim=10000 ,embeddings_dim=100,latent_dim=1024):
    
    input_encoder=layers.Input(shape=(sentence_lenght,) )
    x=layers.Embedding(input_dim=input_dim , output_dim=embeddings_dim, input_length=sentence_lenght,trainable=False,name="Embedding")(input_encoder)
    output_encoder=layers.Bidirectional(   layers.CuDNNLSTM(units=latent_dim,input_shape=(sentence_lenght,embeddings_dim) ,return_sequences=False), merge_mode='sum',name="bi_LSTM")(x)# {'sum', 'mul', 'concat', 'ave', None}.
    return keras.Model(inputs=input_encoder,outputs=output_encoder,name="Text_Encoder")


def get_decoder(sentence_lenght=20,input_dim=10000 ,latent_dim=1024):
    input_decoder=layers.Input(shape=(latent_dim,),name="txt" )
    x = layers.RepeatVector(sentence_lenght)(input_decoder)
    x = layers.Bidirectional(  layers.CuDNNLSTM(units=1000 ,return_sequences=True,input_shape=(sentence_lenght,latent_dim)) , merge_mode='sum',name="bi_LSTM")(x)
    output_decoder= layers.Conv1D(input_dim,kernel_size=1,activation="softmax",name="Conv")(x)
    return keras.Model(inputs=input_decoder,outputs=output_decoder,name="Text_Decoder")
    
def get_training_model(input_shape=(20,),encoder=None,decoder=None):
    model_input=layers.Input(shape=input_shape)
    enc=encoder(model_input)  #(model_input)
    dec=decoder(enc)
        
        
    model=keras.Model(inputs=model_input, outputs=dec)   
    model.compile(loss="sparse_categorical_crossentropy",optimizer=keras.optimizers.Adam(lr=1e-4) )  
    return model
    