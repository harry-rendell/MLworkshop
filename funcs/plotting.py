import matplotlib.pyplot as plt
import numpy as np

def plot_classifications(x_input, y_test, predicted):
    """
    Plot a random set of test data with predicted and true labels on an 8x8 grid

    Parameters
    ----------
    x_train : training images
    y_train : true labels
    predicted : model predictions of x_train

    """
    x_test = np.squeeze(x_input)
    nrows = 8
    ncols = 8
    shuffled_idxs = np.random.randint(0, len(x_test), nrows*ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 16))
    for i, ax in enumerate(axes.ravel()):
        idx = shuffled_idxs[i]
        image = x_test[idx]
        label_true = y_test[idx]
        label_predicted = predicted[idx]
        if label_true != label_predicted:
            # If the true label is different from the prediction of the neural network, plot in red
            ax.imshow(image, cmap=plt.cm.Reds_r, interpolation="nearest")
        else:
            # Otherwise, plot in grayscale
            ax.imshow(image, cmap=plt.cm.Greys_r, interpolation="nearest")
        ax.set_axis_off()
        ax.set_title('predicted: {}\ntrue: {}'.format(label_predicted, label_true))

    plt.subplots_adjust(wspace=1, hspace=1)


def plot_training(history, **kwargs):
    """
    Plot training history with training and test accuracies

    Parameters
    ----------
    history : This is the output from model.fit()
    kwargs : optional keyword arguments to pass to Axes.set()
    """
    fig, axes = plt.subplots(1,2, figsize = (16,5))
    for ax, name in zip(axes, ['loss','accuracy']):
        ax.plot(history.history[name])
        ax.plot(history.history['val_'+name])
        ax.set(title='model '+name, xlabel='epoch', ylabel=name, **kwargs)
    axes[0].legend(['train', 'test'])
    axes[1].legend(['train', 'test'])

def plot_data(x, y):
    """
    Plot a random subset of the data to see what we're working with. Label using the true labels.

    Parameters
    ----------
    x : x_train or x_test
    y : y_train or y_test
    
    """
    if x.shape[-1] == 1:
        x = x.reshape(x.shape[:-1])

    nrows = 8
    ncols = 8
    shuffled_idxs = np.random.randint(0, len(x), nrows*ncols)

    fig, axes = plt.subplots(nrows=6, ncols=6, figsize=(10, 10))
    for i, ax in enumerate(axes.ravel()):
        idx = shuffled_idxs[i]
        image = x[idx]
        label = y[idx]
        
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray, interpolation="nearest")
        ax.set_title("Label: %i" % label)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

def plot_latent_space(latent_output, y_train, n_classes):
    """
    Plot the activations from the 2 node Dense layer for all of the training data.
    Each digit is coloured separately.

    Parameters
    ----------
    latent_output : output from newmodel.predict()
    y_train : true labels
    n_classes : number of digits, in this case 10
    
    """
    # Find the maximum extent of activations which we use to set xlim and ylim for plotting
    xlim, ylim = np.array([latent_output.min(axis=0), latent_output.max(axis=0)]).T 
    
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    for i in range(n_classes):
        mask = y_train == i
        ax.scatter(*latent_output[mask].T, s=0.5, label=i)

    ax.set(xlabel='Activation of node 1', ylabel='Activation of node 2')
    
    lgnd = ax.legend()
    for handle in lgnd.legendHandles:
        handle.set_sizes([100])
