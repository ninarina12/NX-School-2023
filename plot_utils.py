import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_object(obj, label=0):
    fig, ax = plt.subplots(figsize=(3,3))
    
    ax.imshow(obj, origin='lower', cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    
    if label:
        ax.text(0.9, 0.1, 'n = ' + str(label), color='white', ha='right', va='bottom', transform=ax.transAxes)
        
    return fig


def plot_object_and_pattern(obj, diff, label=0):
    fig, ax = plt.subplots(1,3, figsize=(6.25,3), gridspec_kw={'width_ratios': [1,1,0.05]})
    fig.subplots_adjust(wspace=0.1)
    
    cmap = plt.cm.bone
    vmax = 10**np.round(np.log10(diff.max()))
    norm = mpl.colors.LogNorm(vmin=1., vmax=vmax, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    ax[0].imshow(obj, origin='lower', cmap='gray')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    
    if label:
        ax[0].text(0.9, 0.1, 'n = ' + str(label), color='white', ha='right', va='bottom', transform=ax[0].transAxes)

    ax[1].imshow(diff, origin='lower', cmap=cmap, norm=norm)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.colorbar(sm, cax=ax[2])
    return fig


def plot_object_grid(objs, labels=None):
    fig, ax = plt.subplots(4,4, figsize=(8,8))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = ax.ravel()
    
    if len(objs) <= 16:
        indices = np.arange(len(objs))
    else:
        indices = np.random.randint(len(objs), size=16)

    for i, index in enumerate(indices):
        axes[i].imshow(objs[index], origin='lower', cmap='gray')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    for j in range(i + 1,16):
        axes[j].remove()
        
    try: len(labels)
    except: pass
    else:
        for i, index in enumerate(indices):
            axes[i].text(0.9, 0.1, 'n = ' + str(labels[index]), color='white', ha='right', va='bottom',
                         transform=axes[i].transAxes)
    return fig


def plot_pattern_grid(diffs, labels=None):
    fig, ax = plt.subplots(4,5, figsize=(8,8), gridspec_kw={'width_ratios': [1]*4 + [0.05]})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    axes = ax[:,:-1].ravel()

    if len(diffs) <= 16:
        indices = np.arange(len(diffs))
    else:
        indices = np.random.randint(len(diffs), size=16)

    cmap = plt.cm.bone
    vmax = 10**np.round(np.log10(diffs.max()))
    norm = mpl.colors.LogNorm(vmin=1., vmax=vmax, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    for i, index in enumerate(indices):
        axes[i].imshow(diffs[index], origin='lower', cmap=cmap, norm=norm)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    for j in range(i + 1,16):
        axes[j].remove()
        
    for i in range(3):
        ax[i,-1].remove()
    
    try: len(labels)
    except: pass
    else:
        for i, index in enumerate(indices):
            axes[i].text(0.9, 0.1, 'n = ' + str(labels[index]), color='white', ha='right', va='bottom',
                         transform=axes[i].transAxes)
        
    plt.colorbar(sm, cax=ax[-1,-1])
    return fig


def plot_explained_variance(pca):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.scatter(np.arange(pca.n_components_) + 1, np.cumsum(pca.explained_variance_ratio_))
    ax.set_xlabel('Number of components')
    ax.set_ylabel('Cumulative explained variance ratio')
    ax.set_ylim(top=1.)

    print('Explained variance of first 3 components: {:.2f}%'.format(
        np.sum(pca.explained_variance_ratio_[:3]) * 100))
    return fig


def plot_components(pca):
    fig, ax = plt.subplots(4,5, figsize=(8,8), gridspec_kw={'width_ratios': [1]*4 + [0.05]})
    fig.subplots_adjust(hspace=0.15, wspace=0.15)
    axes = ax[:,:-1].ravel()

    comps = pca.components_

    cmap = plt.cm.bwr
    vmax = np.abs(comps).max()
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    N = int(np.sqrt(comps[1].shape))
    for i in range(16):
        axes[i].imshow(comps[i].reshape(N, N), cmap=cmap, norm=norm)
        axes[i].set_title('Component {}'.format(i + 1), fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    for i in range(3):
        ax[i,-1].remove()

    plt.colorbar(sm, cax=ax[-1,-1])
    
    return fig


def plot_inverse_transforms(pca_inverse, diffs):
    fig, ax = plt.subplots(2,5, figsize=(8,4), gridspec_kw={'width_ratios': [1]*4 + [0.05]})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    indices = np.random.randint(len(diffs), size=4)

    cmap = plt.cm.bone
    vmax = 10**np.round(np.log10(diffs.max()))
    norm = mpl.colors.LogNorm(vmin=1., vmax=vmax, clip=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    shape = diffs[0].shape
    for i, index in enumerate(indices):
        ax[0,i].imshow(diffs[index], origin='lower', cmap=cmap, norm=norm)
        ax[0,i].set_xticks([])
        ax[0,i].set_yticks([])
        
        ax[1,i].imshow(pca_inverse[index].reshape(shape), origin='lower', cmap=cmap, norm=norm)
        ax[1,i].set_xticks([])
        ax[1,i].set_yticks([])
    
    ax[0,0].set_ylabel('Original')
    ax[1,0].set_ylabel('Reconstructed')
    ax[0,-1].remove()    
    plt.colorbar(sm, cax=ax[-1,-1])
    return fig


def plot_coefficients(pca_transform, targets):
    fig, ax = plt.subplots(figsize=(8,3))
    p = ax.scatter(pca_transform[:,0], pca_transform[:,1], c=targets + 1, cmap='viridis', ec='black')
    cbar = fig.colorbar(p, ax=ax, ticks=range(1,6), pad=0.01)
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')
    cbar.set_label('Aggregate size')
    return fig


def plot_annotated_coefficients(pca_transform, diffs, targets, x_right=1500, x_left=11000):
    # Broken x-axis plot for better visualization
    fig, ax = plt.subplots(1,2, figsize=(11,4), sharey=True, gridspec_kw={'width_ratios': [1,0.15]})
    fig.subplots_adjust(wspace=0.05)

    # Define colormaps
    cmap1 = plt.cm.viridis
    norm1 = plt.Normalize(vmin=1, vmax=5)
    sm = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
    
    cmap2 = plt.cm.bone
    vmax = 10**np.round(np.log10(diffs.max()))
    norm2 = mpl.colors.LogNorm(vmin=1., vmax=vmax, clip=True)

    # Plot 1-sphere aggregate example
    ax[1].scatter(pca_transform[targets==0][0,0], pca_transform[targets==0][0,1], c=1,
                  cmap=cmap1, norm=norm1, ec='black')
    im = OffsetImage(cmap2(norm2(diffs[targets==0][0])), zoom=0.1)
    im.image.axes = ax[1]

    xy = pca_transform[targets==0][0,:2]
    ab = AnnotationBbox(im, xy, pad=0, bboxprops=dict(ec=cmap1(norm1(1)), lw=2))
    ax[1].add_artist(ab)

    # Define approximate grid for plotting patterns
    n = 10
    x = np.linspace(pca_transform[:,0].min(), x_right, n)
    y = np.linspace(pca_transform[:,1].min(), pca_transform[:,1].max(), n)
    X, Y = np.meshgrid(x, y)
    XY = np.vstack((X.ravel(),Y.ravel())).T
    indices = np.argmin(np.linalg.norm(pca_transform[:,None,:2] - XY[None], axis=-1), axis=0)
    
    # Plot example patterns
    for index in indices:
        ax[0].scatter(pca_transform[index,0], pca_transform[index,1], c=targets[index] + 1,
                      cmap=cmap1, norm=norm1, ec='black')
        im = OffsetImage(cmap2(norm2(diffs[index])), zoom=0.1)
        im.image.axes = ax[0]

        xy = pca_transform[index,:2]
        ab = AnnotationBbox(im, xy, pad=0, bboxprops=dict(ec=cmap1(norm1(targets[index] + 1)), lw=2))
        ax[0].add_artist(ab)

    # Set axis labels
    ax[0].set_ylabel('Second component', fontsize=10)
    fig.supxlabel('First component', fontsize=10)

    # Set axis limits and format spines
    ax[0].set_xlim(right=x_right)
    ax[1].set_xlim(left=x_left)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].yaxis.tick_right()
    ylim = 1.1*np.array(ax[0].get_ylim())

    # Add hatches to top and bottom spines
    d = 0.02
    ax[1].plot((-d - 0.25, d - 0.25), (1 - d, 1 + d), transform=ax[1].transAxes, color='black', clip_on=False)
    ax[1].plot((-d - 0.25, d - 0.25), (-d, d), transform=ax[1].transAxes, color='black', clip_on=False)
    
    ax[1].plot((-d, d), (1 - d, 1 + d), transform=ax[1].transAxes, color='black', clip_on=False)
    ax[1].plot((-d, d), (-d, d), transform=ax[1].transAxes, color='black', clip_on=False)
    ax[0].set_ylim(ylim)
    
    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax[1], ticks=range(1,6), pad=0.1, aspect=36)
    cbar.set_label('Aggregate size')
    return fig


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    fig, ax = plt.subplots(1,2, figsize=(7,3))
    fig.subplots_adjust(wspace=0.3)
    ax[0].plot(range(1, len(train_losses) + 1), train_losses, label='Train.')
    ax[0].plot(range(1, len(val_losses) + 1), val_losses, label='Val.')
    ax[0].set_yscale('log')

    ax[1].plot(range(1, len(train_accuracies) + 1), train_accuracies)
    ax[1].plot(range(1, len(val_accuracies) + 1), val_accuracies)

    ax[0].legend(frameon=False)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    return fig


def plot_classification_statistics(test_probabilities, incorrect, correct):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.boxplot((test_probabilities[incorrect], test_probabilities[correct]))
    ax.set_xticks([1,2])
    ax.set_xticklabels(['Incorrect', 'Correct'])
    ax.set_ylabel('Probability')
    return fig


def plot_confusion_matrix(test_targets, test_predictions):
    cm = confusion_matrix(test_targets, test_predictions, labels=range(5))
    
    fig, ax = plt.subplots(1,2, figsize=(3.15,3), gridspec_kw={'width_ratios': [1,0.05]})
    fig.subplots_adjust(wspace=0.1)

    cmap = plt.cm.bone
    vmax = cm.max()
    norm = plt.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    ax[0].imshow(cm, cmap=cmap, norm=norm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i,j] < 0.3*cm.max():
                color = 'white'
            else:
                color = 'black'
            ax[0].text(j,i, cm[i,j], ha='center', va='center', color=color)

    ax[0].set_xticks(range(5))
    ax[0].set_xticklabels(range(1,6))
    ax[0].set_yticks(range(5))
    ax[0].set_yticklabels(range(1,6))
    ax[0].set_xlabel('Predicted aggregate size')
    ax[0].set_ylabel('True aggregate size')
    
    plt.colorbar(sm, cax=ax[1])
    return fig