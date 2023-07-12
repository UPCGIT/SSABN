import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras import callbacks
from scipy import io as sio
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import h5py


class HSI:
    '''
    A class for Hyperspectral Image (HSI) data.
    '''
    def __init__(self,data, rows, cols, gt,sgt,patch_size):
        if data.shape[0] < data.shape[1]:
            data = data.transpose()

        self.bands = np.min(data.shape)
        self.rows=rows
        self.cols=cols
        self.p=gt.shape[0]

        ####padding
        image = np.reshape(data, (self.rows, self.cols, self.bands))
        h=rows
        w=cols
        h1=h//patch_size if h//patch_size==0 else h // patch_size + 1
        w1=w//patch_size if w//patch_size==0 else w//patch_size+1
        image_pad = np.pad(image, ((0, patch_size * h1 - h), (0, patch_size * w1 - w), (0, 0)),'edge')
      
        self.prows = image_pad.shape[0]
        self.pcols = image_pad.shape[1]
        self.image_pad = image_pad
        self.image = image

        self.gt = gt
        self.sgt=sgt
    
    def array(self,n):
        """this returns a array of spectra with shape num pixels x num bands
        
        Returns:
            a matrix -- array of spectra
        """
        if n==1:
            return np.reshape(self.image_pad,(self.prows*self.pcols,self.bands))
        else:
            return np.reshape(self.image, (self.rows * self.cols, self.bands))
    
    def get_bands(self, bands):
        return self.image[:,:,bands]

    def crop_image(self,start_x,start_y,delta_x=None,delta_y=None):
        if delta_x is None: delta_x = self.cols - start_x
        if delta_y is None: delta_y = self.rows - start_y
        self.cols = delta_x
        self.rows = delta_y
        self.image = self.image[start_x:delta_x+start_x,start_y:delta_y+start_y,:]
        return self.image


def load_HSI(path,patch_size=4):
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = h5py.File(path, 'r')
    
    numpy_array = np.asarray(data['Y'], dtype=np.float32)
    numpy_array = numpy_array / np.max(numpy_array.flatten())
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()
    
    if 'GT' in data.keys():
        gt = np.asarray(data['GT'], dtype=np.float32)
    else:
        gt = None
    if 'S_GT' in data.keys():
        sgt = np.asarray(data['S_GT'], dtype=np.float32)
    else:
        sgt = None
    return HSI(numpy_array, n_rows, n_cols, gt,sgt,patch_size)


def numpy_SAD(y_true, y_pred):
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred))
    if cos>1.0: cos = 1.0
    return np.arccos(cos)


def order_endmembers(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    mydict = {}
    sad_mat = np.ones((num_endmembers, num_endmembers))
    #for i in range(num_endmembers):
        #endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        #endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()
    for i in range(num_endmembers):
        for j in range(num_endmembers):
            sad_mat[i, j] = numpy_SAD(endmembers[i, :], endmembersGT[j, :])
    rows = 0
    while rows < num_endmembers:
        minimum = sad_mat.min()
        index_arr = np.where(sad_mat == minimum)
        if len(index_arr) < 2:
            break
        index = (index_arr[0][0], index_arr[1][0])
        if index[0] in mydict.keys():
            sad_mat[index[0], index[1]] = 100
        elif index[1] in mydict.values():
            sad_mat[index[0], index[1]] = 100
        else:
            mydict[index[0]] = index[1]
            sad_mat[index[0], index[1]] = 100
            rows += 1
    ASAM = 0
    num = 0
    for i in range(num_endmembers):
        if np.var(endmembersGT[mydict[i]]) > 0:
            ASAM = ASAM + numpy_SAD(endmembers[i, :], endmembersGT[mydict[i]])
            num += 1

    return mydict, ASAM / float(num)

def plotEndmembersAndGT(endmembers, endmembersGT):
    num_endmembers = endmembers.shape[0]
    n = int(num_endmembers // 2)
    if num_endmembers % 2 != 0:
        n = n + 1
        
    hat, sad = order_endmembers(endmembersGT, endmembers)
    fig = plt.figure(num=1, figsize=(8, 8))
    plt.clf()
    title = "mSAD: " + format(sad, '.3f') + " radians"
    st = plt.suptitle(title)
    
    for i in range(num_endmembers):
        endmembers[i, :] = endmembers[i, :] / endmembers[i, :].max()
        endmembersGT[i, :] = endmembersGT[i, :] / endmembersGT[i, :].max()

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(endmembers[hat[i], :], 'r', linewidth=1.0)
        plt.plot(endmembersGT[i, :], 'k', linewidth=1.0)
        plt.ylim((0,1))
        ax.set_title(format(numpy_SAD(endmembers[hat[i], :], endmembersGT[i, :]), '.3f'))
        ax.get_xaxis().set_visible(False)

    plt.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.88)
    plt.draw()
    plt.pause(0.001)
    
def plotAbundancesSimple(abundances,name):
    abundances = np.transpose(abundances, axes=[1, 0, 2])
    num_endmembers = abundances.shape[2]
    n = num_endmembers // 2
    if num_endmembers % 2 != 0: n = n + 1
    cmap='jet'
    fig =plt.figure(figsize=[12, 12])
    rect = fig.patch
    rect.set_facecolor('white')
    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position='bottom', size='5%', pad=0.05)
        im = ax.imshow(abundances[:, :, i], cmap=cmap)
        plt.colorbar(im, cax=cax, orientation='horizontal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        im.set_clim(0, 1)
        
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
    fig.savefig(name+'.png')
    plt.close()

class PlotWhileTraining(callbacks.Callback):
    def __init__(self, plot_every_n, hsi):
        super(PlotWhileTraining, self).__init__()
        self.plot_every_n = plot_every_n
        self.input = hsi.array(n=1)
        self.cols = hsi.cols
        self.rows = hsi.rows
        self.endmembersGT = hsi.gt
        self.sads = None
        self.epochs = []

    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []
        self.sads = []

    def on_batch_end(self, batch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('SAD'))
        self.num_epochs = epoch
        endmembers = self.model.layers[-1].get_weights()[0]
        endmembers = np.squeeze(endmembers)
       
        if self.plot_every_n == 0 or epoch % self.plot_every_n != 0:
            return
        if self.endmembersGT is not None:
            plotEndmembersAndGT(self.endmembersGT, endmembers)
        else:
            plotEndmembers(endmembers)

def reconstruct(A,S):
    s_shape = S.shape
    S = np.reshape(S,(S.shape[0]*S.shape[1],S.shape[2]))
    reconstructed = np.matmul(S,A)
    reconstructed = np.reshape(reconstructed, (s_shape[0], s_shape[1],reconstructed.shape[1]))
    return reconstructed

def compute_sad(gt,A):
    hat, sad = order_endmembers(gt,A)
    num_endmembers=A.shape[0]
    sad_mat = [0]*num_endmembers
    for i in range(num_endmembers):
        sad_mat[i]=numpy_SAD(A[hat[i], :], gt[i, :])
    return sad_mat,sad
