import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler , MinMaxScaler, normalize, QuantileTransformer

from numpy import hstack
from numpy import zeros, ones
from numpy.random import rand
from numpy.random import randn

from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot
import numpy as np
from keras.layers import Dense, Activation, ActivityRegularization, Dropout, BatchNormalization
from keras import regularizers


from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras import backend, Model
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import BatchNormalization, Concatenate, LeakyReLU, Flatten, Reshape, Dense, Input, Conv1DTranspose, Conv1D
from keras.initializers import RandomNormal
from keras.constraints import Constraint
from matplotlib import pyplot
import random
 
# clip model weights to a given hypercube
class ClipConstraint(Constraint):
    # set clip value when initialized
    def __init__(self, clip_value):
        self.clip_value = clip_value
 
    # clip model weights to hypercube
    def __call__(self, weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)
 
    # get the config
    def get_config(self):
        return {'clip_value': self.clip_value}

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)
 
def define_critic(in_shape=(None,40)):    
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # weight constraint
    const = ClipConstraint(0.01)
    # define model
    inp = Input(shape=in_shape)
    
    x = Dense(32, kernel_initializer=init, kernel_constraint=const)(inp)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate = 0.2)(x)
    
    x = Dense(32,  kernel_initializer=init, kernel_constraint=const)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate = 0.2)(x)
    
    x = Dense(16, kernel_initializer=init, kernel_constraint=const)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(rate = 0.2)(x)
    
    x = Dense(1)(x)
    model = Model(inputs=inp, outputs=x)
    opt = RMSprop(lr=0.0005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model
 
# define the standalone generator model
def define_generator(latent_dim):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    n_nodes = 16
    in1 = Input(shape=(latent_dim,))
    
    x1 = Dense(n_nodes, kernel_initializer=init)(in1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    x1 = Reshape((n_nodes,1))(x1)

    x1 = Conv1DTranspose(32, kernel_size=4, strides=2)(x1)
    x1 = BatchNormalization()(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)
    
    x1 = Conv1D(1, 7, activation='tanh', padding='same', kernel_initializer=init)(x1)
    x1 = Flatten()(x1)

    
    in2 = Input(shape=(1,))
    
    x2 = Dense(2, kernel_initializer=init)(in2)
    x2 = LeakyReLU(alpha=0.2)(x2)
    x2 = Dropout(rate = 0.5)(x2)
    
    x2 = Dense(4, kernel_initializer=init)(x2)
    x2 = BatchNormalization()(x2)
    x2 = LeakyReLU(alpha=0.2)(x2)
    x2 = Dropout(rate = 0.5)(x2)
    
    x2 = Dense(6, activation='softmax', kernel_initializer=init)(x2)
    
    output = Concatenate()([x1,x2])
    model = Model(inputs=[in1, in2], outputs=output)    
    return model


# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
    # make weights in the critic not trainable
    for layer in critic.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # connect them
    in1 = Input(shape=(latent_dim,))
    in2 = Input(shape=(1,))
    out1 = generator([in1, in2])
    out2 = critic(out1)
    model = Model([in1, in2], out2)
    opt = RMSprop(lr=0.0005)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model

 
# select real samples
def generate_real_samples(dataset, n_samples):
    
    a = random.sample(range(len(dataset)),n_samples)
    X = dataset.iloc[a,:]
    # generate class labels, -1 for 'real'
    y = -ones((n_samples, 1))
    return X, y
 
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input
 
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    x_input1 = generate_latent_points(latent_dim, n_samples)
    x_input2 = generate_latent_points(1,n_samples)
    X = generator.predict([x_input1,x_input2])
    # create class labels with 1.0 for 'fake'
    y = ones((n_samples, 1))
    return X, y
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(10 * 10):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # save plot to file
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))

# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
    # plot history
    pyplot.plot(d1_hist, label='crit_real')
    pyplot.plot(d2_hist, label='crit_fake')
    pyplot.plot(g_hist, label='gen')
    pyplot.legend()
    pyplot.savefig('plot_line_plot_loss2.png')
    pyplot.show()

# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=150, n_batch= 182, n_critic=5):
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset.shape[0] / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    # lists for keeping track of loss
    c1_hist, c2_hist, g_hist = list(), list(), list()
    # manually enumerate epochs
    for i in range(n_steps):
        # update the critic more than the generator
        c1_tmp, c2_tmp = list(), list()
        for _ in range(n_critic):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
#             c_loss1 = c_model.train_on_batch(X_real, y_real)
#             c1_tmp.append(c_loss1)
            # generate 'fake' examples
            X_mix = np.stack((X_real, X_fake))
            y_mix = np.stack((y_real, y_fake))
            c_loss = c_model.train_on_batch(X_mix, y_mix)
#             c_loss2 = c_model.train_on_batch(X_fake, y_fake)
#             c2_tmp.append(c_loss2)
            c2_tmp.append(c_loss)
        # store critic loss
        c1_hist.append(mean(c1_tmp))
        c2_hist.append(mean(c2_tmp))
        # prepare points in latent space as input for the generator
        X_gan1 = generate_latent_points(latent_dim, n_batch)
        X_gan2 = generate_latent_points(1, n_batch)
        # create inverted labels for the fake samples
        y_gan = -ones((n_batch, 1))
        # update the generator via the critic's error
        g_loss = gan_model.train_on_batch([X_gan1, X_gan2], y_gan)
        g_hist.append(g_loss)
        # summarize loss on this batch
        print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
        # evaluate the model performance every 'epoch'
        # if (i+1) % bat_per_epo == 0:
              #  summarize_performance(i, g_model, latent_dim)
    # line plots of loss
    plot_history(c1_hist, c2_hist, g_hist)


# size of the latent space
latent_dim = 5
# create the critic
critic = define_critic()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, critic)
# train model
gan_model.summary()


scaler = MinMaxScaler()

x_t = scaler.fit_transform(X)
x_t = pd.DataFrame(x_t, columns = X.columns)

train(generator, critic, gan_model, x_t, latent_dim)



half_batch = 1000
x_fake, y_fake = generate_fake_samples(generator, latent_dim, half_batch)
df_fake = pd.DataFrame(x_fake, columns=X.columns)



def normalize_chemicals(df):

    df[chemicals] = df[chemicals].multiply(pd.DataFrame(100/df[chemicals].sum(axis=1)).loc[:, 0], axis='index')

    return df


def normalize_categoricals(df):
    categoricals = ['Extruded', 'ECAP', 'Cast slow', 'Cast fast', 'Cast HT', 'Wrought']
    df_temp = df.copy()[categoricals]
    for ind, row in df_temp.iterrows():
        ls = [0,0,0,0,0,0]
        ls[np.argmax(row)] = 1
        df_temp.iloc[ind,:] = ls
    df[categoricals] = df_temp
    return df


def replace_negatives(x):
    if x < 0:
        return 0
    else:
        return x

    
df_target = scaler.inverse_transform(df_fake)
df = df_target.applymap(replace_negatives)
df = normalize_chemicals(df)
df = normalize_categoricals(df)


