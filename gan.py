import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from PIL import Image, TarIO

from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, Input
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

from jpg_to_csv import read_from_csv

def ones_target(size):
    np.ones(size=(size,1))
def zeros_target(size):
    return np.zeros(size=(size,1))
def noise(size):
    return np.random.normal(size=(size,100))

#n is the size of array, prob 125 x 125

#n: num of pixels in image
def define_discriminator(n=125 ** 2):
    a = .2
    a2 = .3

    model = Sequential()
   
    model.add(Dense(2**13,input_dim=(n)))
    model.add(LeakyReLU(alpha=a))
    model.add(Dropout(rate=a2))

    model.add(Dense(2**11, input_dim=(2**13)))
    model.add(LeakyReLU(alpha=a))
    model.add(Dropout(rate=a2))

    model.add(Dense(2**8, input_dim=(2**11)))
    model.add(LeakyReLU(alpha=a))
    model.add(Dropout(rate=a2))

    model.add(Dense(1, input_dim=(2 ** 8), activation='sigmoid'))

    # compile model
    opt = Adam(lr=0.01, beta_1=.5)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

#n: num of pixels in image
def define_generator(n=125 ** 2):
    a = .2
    a2 = .3
    n_out = n
    n_features = 100

    model = Sequential()
    
    model.add(Dense(2**8, input_dim=(n_features)))
    model.add(LeakyReLU(alpha=a))

    model.add(Dense(2**10))
    model.add(LeakyReLU(alpha=a))

    model.add(Dense(2**12))
    model.add(LeakyReLU(alpha=a))

    model.add(Dense(2**14))
    model.add(LeakyReLU(alpha=a))

    model.add(Dense(n_out, activation='tanh'))
    
    opt = Adam(lr=0.01,beta_1=.5)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def define_gan(disc, gen):

    disc.trainable = False
    # gan_input = noise(1)
    gan_input = Input(shape=(100,))
    x = gen(gan_input)
    gan_output = disc(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

#epoch: which training epoch
#generator: gen model
#examples: num ex to generate
def plot_generated_images(epoch, generator, outputFolderName, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 125, 125)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(outputFolderName + '/gan_generated_image %d.png' % epoch)

#N: num of images
#size: num of pixels in each image
#myDir: location of csv containing rows of pixels
def load_data(myDir):
    # N = 200
    # myDir = 'img_pixels.csv'
    # size = 125
    data = read_from_csv(myDir)
    return data

#fileLocation: location to read in pixels
#outputFolderName: folder to output progression of predictions during training
def training(fileLocation, outputFolderName, saveModelAsName, epochs=1, batch_size=40):

    #Loading the data
    # (X_train, y_train, X_test, y_test) = load_data()
    # batch_count = X_train.shape[0] / batch_size

    X_train = load_data(fileLocation)
    size = X_train.shape[1]
    # Creating GAN
    generator = define_generator(size)
    discriminator = define_discriminator(size)
    gan = define_gan(discriminator, generator)

    for e in range(1, epochs+1):
        print("Epoch %d" % e)
        for _ in tqdm(range(batch_size)):
            #generate  random noise as an input  to  initialize the  generator
            noise = np.random.normal(0, 1, [batch_size, 100])

            # Generate fake MNIST images from noised input
            generated_images = generator.predict(noise)

            # Get a random set of  real images
            image_batch = X_train[np.random.randint(
                low=0, high=X_train.shape[0], size=batch_size)]

            #Construct different batches of  real and fake data
            X = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9

            #Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            #Tricking the noised input of the Generator as real data
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)

            # During the training of gan,
            # the weights of discriminator should be fixed.
            #We can enforce that by setting the trainable flag
            discriminator.trainable = False

            #training  the GAN by alternating the training of the Discriminator
            #and training the chained GAN model with Discriminator’s weights freezed.
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:

            plot_generated_images(e, generator, outputFolderName)
    
    generator.save('models/generator_' + saveModelAsName)
    discriminator.save('models/discriminator_' + saveModelAsName)
    gan.save('models/gan_' + saveModelAsName)

# N = 200
# someLoc = 'img_pixels.csv'
# size = 125

# data = read_from_csv(someLoc,N,size)


# gen = define_generator()
# disc = define_discriminator()
# gan = define_gan(disc, gen)

# gen.summary()
# disc.summary()
# gan.summary()

# epochs = 1
# batch_size = 40

# training(epochs, batch_size)




# print('done')

