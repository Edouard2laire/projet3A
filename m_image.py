from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
from utils import compute_vgg_output,ImageSequence,l1_l2

def get_encoder(input_shape=(64,64,3),latent_dim=1024,number_of_block=3):
    
    input_encoder=layers.Input(shape=input_shape,name="image")
    vgg = VGG16(weights=None, include_top=False,input_tensor=input_encoder, input_shape=(64,64,3),pooling='none')
    x = vgg.get_layer("block{}_pool".format(number_of_block)).output
    x= layers.Flatten()(x)
    output_VGG=layers.Dense(latent_dim,name="Dense")(x)
    encoder= keras.Model(inputs=input_encoder, outputs=output_VGG, name='Image_Encoder');
    return encoder
    
def get_decoder(latent_dim=1024,input_shape=(8,8,256),number_of_block=3):
    input_decoder= layers.Input(shape=(latent_dim,))
    x=layers.Dense(np.prod(input_shape),name="Dense")(input_decoder)
    x= layers.Reshape(input_shape)(x)
    
    if number_of_block>= 4:
        x = layers.UpSampling2D((2, 2),name="block4_up")(x)
        x= layers.Conv2D( 512, 3, strides=(1, 1), padding='same',activation='relu',name="block4_conv1")(x)
        x= layers.Conv2D( 512, 3, strides=(1, 1), padding='same',activation='relu',name="block4_conv2")(x)
        x= layers.Conv2D( 256, 3, strides=(1, 1), padding='same',activation='relu',name="block4_conv3")(x)
        
    if number_of_block>= 3:
        x = layers.UpSampling2D((2, 2),name="block3_up")(x)
        x= layers.Conv2D( 256, 3, strides=(1, 1), padding='same',activation='relu',name="block3_conv1")(x)
        x= layers.Conv2D( 256, 3, strides=(1, 1), padding='same',activation='relu',name="block3_conv2")(x)
        x= layers.Conv2D( 128, 3, strides=(1, 1), padding='same',activation='relu',name="block3_conv3")(x)
    
    if number_of_block>= 2:
        x = layers.UpSampling2D((2, 2),name="block2_up")(x)
        x= layers.Conv2D( 128, 3, strides=(1, 1), padding='same',activation='relu',name="block2_conv1")(x)
        x= layers.Conv2D( 64, 3, strides=(1, 1), padding='same',activation='relu',name="block2_conv2")(x)
    
    if number_of_block>= 1:
        x = layers.UpSampling2D((2, 2),name="block1_up")(x)
        x= layers.Conv2D( 64, 3, strides=(1, 1), padding='same',activation='relu',name="block1_conv1")(x)
        x= layers.Conv2D( 64, 3, strides=(1, 1), padding='same',activation='relu',name="block1_conv2")(x)
    
    output= layers.Conv2D( 3, 3, strides=(1, 1), padding='same',activation='sigmoid',name="image")(x)
    decoder = keras.Model(inputs=input_decoder, outputs=output, name='Image_Decoder');
    return decoder


    
def get_training_model(input_shape=(64,64,3),encoder=None,decoder=None,training_config=dict()):

    
    
    model_outputs=[]
    model_loss=[]
    loss_weights=[]
    expected_value=tuple()
    
    
    model_input=layers.Input(shape=input_shape,name="model_input")
    enc=encoder(model_input)  #(model_input)
    dec=decoder(enc)
    
    
    model=keras.Model(inputs=model_input, outputs=dec)
    #model.compile(loss="mse",optimizer=keras.optimizers.Adam(lr=1e-4) ) 
    #model.summary()
    
    
    model_outputs.append(dec)
    model_loss.append(l1_l2)
    loss_weights.append(1)
    
    
    if "vgg_loss" in training_config and training_config["vgg_loss"]:
        
        # Definition du vgg-pretrained
        input_vgg=layers.Input(shape=input_shape)
        output_vgg=compute_vgg_output(input_shape=input_shape,output_layer="block1_pool",name='block1_vgg')(input_vgg)
        VGG=keras.Model(inputs=input_vgg, outputs=output_vgg,name="vgg_block")
    
        # Appel de VGG
        dec_vgg=VGG(dec)
        out_ivgg=VGG(model_input)
        
        # Calcul de l'erreur
        subtraction=layers.Subtract()([dec_vgg,out_ivgg])
        
        model_outputs.append(subtraction)
        model_loss.append("mse")
        loss_weights.append(1e-3)
        expected_value += (np.zeros( (32,*subtraction.get_shape().as_list()[1:])) ,)
        
    if "intermediate_connection" in training_config : 
        #not working        
        for source,dest in training_config["intermediate_connection"]:
            
            source_layer=model.get_layer("Encoder").get_layer(source).output
            dest_layer= model.get_layer("Decoder").get_layer(dest).input
            
            subtraction=layers.Subtract()([source_layer,dest_layer])
            
            model_outputs.append(subtraction)
            model_loss.append("mse")
            loss_weights.append(1e-3)
            expected_value += (np.zeros( (32,*subtraction.get_shape().as_list()[1:])) ,)

    M4 = keras.Model(inputs=model_input, outputs=model_outputs);
    M4.compile(loss=model_loss,loss_weights=loss_weights,optimizer=keras.optimizers.Adam(lr=1e-4) ) 
    
    return (M4,expected_value)
    
    
    
    