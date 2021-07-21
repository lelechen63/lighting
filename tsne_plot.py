import os
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')


# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy
RS = 20150101

def scatter(x, colors):
    length = np.unique(colors).shape[0]
    print (length)
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", length))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    txts = []
    for i in range( length):
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


def demo():
    digits = load_digits()
    X = np.vstack([digits.data[digits.target==i]
                for i in range(10)])
    y = np.hstack([digits.target[digits.target==i]
                for i in range(10)])
   
    digits_proj = TSNE(random_state=RS).fit_transform(X)
    scatter(digits_proj, y)
    plt.savefig('digits_tsne-generated.png', dpi=120)

def code_vis(type = 1):
    code_path = '/data/home/us000042/lelechen/data/Facescape/reg_code'
    pids = os.listdir(code_path)
    Y = []
    X = []
    # if type = 1, Y would be pid, else it would be exp
    for pid in pids:
        for exp in os.listdir( os.path.join( code_path, pid ) ):
            code_p = os.path.join( code_path, pid, exp )
            X.append(np.load(code_p))
            if type ==1:
                Y.append(int(pid) -1)
            else:
                exp_id = exp.split('_')[0]
                Y.append(int(exp_id) -1)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print (X.shape)
    print (Y.shape)
    digits_proj = TSNE(random_state=RS).fit_transform(X)
    print (digits_proj.shape)
    scatter(digits_proj, Y )
    plt.savefig('%d.png'%type, dpi=120)

code_vis(0)       