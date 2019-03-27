import tensorflow.keras.layers as layers
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import *
import tensorflow.keras.losses as losses

from glove import Corpus, Glove


## Image

#class SSR(Layer):
#    def __init__(self, epsilon=1e-12, **kwargs):
#        super(SSR, self).__init__(**kwargs)
#        self.epsilon = epsilon
#
#   def build(self, input_shape):
#        super(SSR, self).build(input_shape)
#
#    def call(self, inputs, **kwargs):
#        return K.sign(inputs) * K.sqrt(K.abs(inputs) + self.epsilon)

#    def compute_output_shape(self, input_shape):
#        return input_shape

#    def get_config(self):
#        base_config = super(SSR, self).get_config()
#        config = {}
#        return dict(list(base_config.items()) + list(config.items()))


class L2Normalisation(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(L2Normalisation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(L2Normalisation, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.math.l2_normalize(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super(L2Normalisation, self).get_config()
        config = {"axis": self.axis}
        return dict(list(base_config.items()) + list(config.items()))


class compute_vgg_output(layers.Layer):
    def __init__(self,input_shape,output_layer,**kwargs):

        vgg_input = layers.Input(shape=(64,64,3))
        model = VGG16(weights='imagenet', input_tensor=vgg_input,include_top=False, input_shape=input_shape)
        vgg_output = model.get_layer(output_layer).output

        vgg = keras.Model(inputs=vgg_input, outputs=vgg_output)
        for layer in vgg.layers:
            layer.trainable = False

        self.vgg = vgg
        super(compute_vgg_output, self).__init__(**kwargs)

    def build(self, input_shape):
        super(compute_vgg_output, self).build(input_shape)

    def call(self, inputs, **kwargs):
        #if  tf.contrib.framework.is_tensor(inputs):
        return self.vgg(to_vgg(inputs,is_tensor=True))
        #return self.vgg(to_vgg(inputs,is_tensor=False))

    def compute_output_shape(self, input_shape):
        return self.vgg.output.shape

    def get_config(self):
        return super(compute_vgg_output, self).get_config()

# https://stackoverflow.com/questions/46622428/what-is-the-expected-input-range-for-working-with-keras-vgg-models
def to_vgg(image,is_tensor=False):
    image=image[...,::-1]*255
    
    if is_tensor : 
        mean = [[[[103.939, 116.779, 123.68]]]] # 1 x 1 x 1 x 3
        return tf.math.subtract( image ,mean )
    else : 
        image[...,0]-= 103.939
        image[...,1]-= 116.779
        image[...,2]-= 123.68

        return image
        

def from_vgg(image):
    mean = [103.939, 116.779, 123.68]

    image[..., 0] += mean[0]
    image[..., 1] += mean[1]
    image[..., 2] += mean[2]

    image= np.clip(image[:,:,::-1],0,255)
    return image

def l1_l2(x,y):
    return losses.mean_absolute_error(x,y) + losses.mean_squared_error(x,y)
    

class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, path, img_db, img_shape=(64, 64), batch_size=32, vgg_loss=False):
        self.path = path
        self.img_db = img_db
        self.img_shape = img_shape
        self.batch_size = batch_size

        self.vgg_loss = vgg_loss
        self.expected_value=tuple()

    def set_expected_value(self,expected):
        self.expected_value=expected
        
    def __len__(self):
        return len(self.img_db) // self.batch_size

    def __getitem__(self, idx):
        batch = np.zeros((self.batch_size, *self.img_shape,3))

        for i, name in enumerate(self.img_db[idx*self.batch_size:(idx+1)*self.batch_size]):
            image = load_img(self.path + name, target_size=self.img_shape)
            batch[i, :, :, :] = np.array(image)
            
        batch/=255 
        
        if len(self.expected_value) > 0 : 
            return batch, (batch,*self.expected_value)
        return batch, batch

## Text 

class TextSequence(tf.keras.utils.Sequence):
    def __init__(self, sentences,glove,input_dim=400000, max_len=60, batch_size=32):
        self.sentences = sentences
        self.max_len = max_len
        self.input_dim=input_dim
        self.glove=glove
        self.batch_size=batch_size
               
    def __len__(self):
        return len(self.sentences) // self.batch_size

    def __getitem__(self, idx):
        sentences=self.sentences[idx*self.batch_size:(idx+1)*self.batch_size]
        batch=encode_sentences(sentences, glove=self.glove,maxlen=self.max_len,num_classes=self.input_dim)
        return batch, batch
        
        
def encode_sentences(sentences, glove,maxlen=20,num_classes=400000):
  
    output=np.full((len(sentences),maxlen),0)
    unk= glove.dictionary["<unk>"] 

  
    for i,sentence in enumerate(sentences) : 
        for j,word in enumerate(sentence) : 
        
            if j < maxlen : #ttruncate large sentence
                pos=glove.dictionary.get(word)
                if pos is None : 
                    pos=unk
                    
                output[i,j]=pos 
    return output


def decode_sentence(data, glove):
    if len(data.shape) >= 2:
        data= np.argmax(data, axis=1)
    
    sentence=[]
  
    for word_pos in data : 
        word=glove.inverse_dictionary.get(word_pos)
        sentence.append( "<unk>" if word is None else word ) 
      
    return sentence
  
  
def analyze_corpus(corpus,n_word=10000):
    occr=dict(); # Nombre d'occurence de chaque mot
    occr[" "]=0;
    occr["<unk>"]=0;
    N=0; # Nombre de mot dans le dictionaire
    D=dict();
    
    # On compte le nombre d'occurence de chaque mot
    for  headline in corpus[:] :
        for word in headline:
            N+=1;
            try:
                occr[word] += 1
            except KeyError:
                occr[word] = 1
            
    print(f'{N} words analyzed');
    print('\n10 mosts common words : ')
    # On donne un identifiant aux 10k mots les plus fréquents
    
    D[" "]=0; 
    D["<unk>"]=1; # 1 = mot inconnu
    c=2; 
    for w in sorted(occr, key=occr.get, reverse=True):
        D[w]=c;
        c+=1;
        if c >= n_word: #max : 10383
            break;
    
    # Affichage des 15 premiers mots
    c=0;
    for keys in D : 
        print( keys + ' : ' + str( D[keys]) + '( '+ str(occr[keys])+' occurences dans le dataset)')
        c+=1;
        if c >= 10 :
            break;
    print('\n{} of {} unique words stored'.format(len(D),len(occr)) );        
    return D
    
def create_embedding(glove,corpus,n_word=10000):
    
    D=analyze_corpus(corpus,n_word)
        
    sub_dictionary={}
    sub_inverse_dictionary={}
    embedding_matrix = np.zeros((len(D), glove.no_components))
    
    unk=np.mean(glove.word_vectors,axis=0)
    
    k=0;
    for word, i in D.items():
        sub_inverse_dictionary[i]=word
        sub_dictionary[word]=i
        
        pos=glove.dictionary.get(word)
        if pos is not None :
            embedding_matrix[i,:]=glove.word_vectors[pos]
        elif i > 0 :
            print(word,end=', ')
            embedding_matrix[i,:]=unk
            k+=1
    
    print('\n{} word are not embedded'.format(k))
    print('Embedding matrix shape ',embedding_matrix.shape)

    sub_glove = Glove(no_components=glove.no_components, learning_rate=0.05)
    sub_glove.dictionary=sub_dictionary
    sub_glove.inverse_dictionary=sub_inverse_dictionary
    sub_glove.word_vectors=embedding_matrix    
    
    return sub_glove