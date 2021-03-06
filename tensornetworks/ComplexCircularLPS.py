# -*- coding: utf-8 -*-

from .CircularMPSClass import TN
import numpy as np
"""from sklearn.externals.six.moves import xrange"""
from six import StringIO
from past.builtins import xrange


class ComplexCircularLPS(TN):
    """Locally purified states (LPS) with complex elements
    Parameters
    ----------
    D : int, optional
        Rank/Bond dimension of the MPS
    learning_rate : float, optional
        Learning rate of the gradient descent algorithm
    batch_size : int, optional
        Number of examples per minibatch.
    n_iter : int, optional
        Number of iterations (epochs) over the training dataset to perform
        during training.
    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator and of the initial parameters.
        If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.
    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.
    mu : int, optional
        Dimension of the purification link
    ----------
    Attributes
    ----------
    w : numpy array, shape (m_parameters)
        Parameters of the tensor network
    norm : float
        normalization constant for the probability distribution
    n_samples : int
        number of training samples
    n_features : int
        number of features in the dataset
    d : int
        physical dimension (dimension of the features)
    m_parameters : int
        number of parameters in the network
    history : list
        saves the training accuracies during training
    """

    def __init__(self, D=4, learning_rate=0.1, batch_size=10,
                 n_iter=100, random_state=None, verbose=False, mu=2):
        self.D = D
        self.learning_rate = float(learning_rate)
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        self.mu = mu

    def _probability(self, x):
        """Unnormalized probability of one configuration P(x)
        Parameters
        ----------
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        probability : float
        """
        w2 = np.reshape(
            self.w, (self.n_features, self.d, self.D, self.D, self.mu))

        tmp2 = np.einsum('imj,klj->ikml', w2[0, x[0], :, :, :], np.conjugate(
            w2[0, x[0], :, :, :])).reshape((self.D*self.D, self.D*self.D))
        # ~ tmp1 = np.einsum('imj,klj->ikml',w2[self.n_features-1,x[self.n_features-1],:,:,:],np.conjugate(w2[self.n_features-1,x[self.n_features-1],:,:,:])).reshape((self.D*self.D,self.D*self.D))
        # ~ tmp2 = np.dot(tmp1,tmp)
        for i in xrange(1, self.n_features):
            tmp = np.einsum('imj,klj->ikml', w2[i, x[i], :, :, :],
                            np.conjugate(w2[i, x[i], :, :, :])).reshape((self.D*self.D, self.D*self.D))
            tmp2 = np.dot(tmp2, tmp)

        probability = np.abs(np.trace(tmp2))
        return probability

    def _computenorm(self):
        """Compute norm of probability distribution
        Returns
        -------
        norm : float
        """
        w2 = np.reshape(
            self.w, (self.n_features, self.d, self.D, self.D, self.mu))

        tmp2 = np.einsum('pimj,pklj->ikml', w2[0, :, :, :, :],
                         np.conj(w2[0, :, :, :, :])).reshape((self.D*self.D, self.D*self.D))
        # ~ tmp1 = np.einsum('pimj,pklj->ikml',w2[self.n_features-1,:,:,:,:],
        # ~ np.conjugate(w2[self.n_features-1,:,:,:,:])).reshape((self.D*self.D,self.D*self.D))
        # ~ tmp2 = np.dot(tmp1,tmp)
        for i in xrange(1, self.n_features):
            tmp = np.einsum('pimj,pklj->ikml', w2[i, :, :, :, :],
                            np.conjugate(w2[i, :, :, :, :])).reshape((self.D*self.D, self.D*self.D))
            tmp2 = np.dot(tmp2, tmp)

        norm = np.abs(np.trace(tmp2))
        return norm

    def _derivative(self, x):
        """Compute the derivative of P(x)
        Parameters
        ----------
        x : numpy array, shape (n_features,)
            One configuration
        Returns
        -------
        derivative : numpy array, shape (m_parameters,)
        """
        w2 = np.reshape(
            self.w, (self.n_features, self.d, self.D, self.D, self.mu))
        derivative = np.zeros((self.n_features, self.d, self.D, self.D, self.mu),
                              dtype=np.complex128)

        # Store intermediate tensor contractions for the derivatives:
        # left to right and right to left
        # tmp stores the contraction of the first i+1 tensors from the left
        # in tmp[i,:,:], tmp2 the remaining tensors on the right
        # the mps contracted is the remaining contraction tmp[i-1]w[i]tmp2[i+1]
        tmp = np.zeros((self.n_features, self.D*self.D,
                        self.D*self.D), dtype=np.complex128)
        tmp2 = np.zeros((self.n_features, self.D*self.D,
                         self.D*self.D), dtype=np.complex128)
        tmp0 = np.einsum('imj,klj->ikml', w2[0, x[0], :, :, :],
                         np.conjugate(w2[0, x[0], :, :, :])).reshape((self.D*self.D, self.D*self.D))
        tmp1 = np.einsum('imj,klj->ikml', w2[self.n_features-1, x[self.n_features-1], :, :, :],
                         np.conjugate(w2[self.n_features-1, x[self.n_features-1], :, :, :])).reshape((self.D*self.D, self.D*self.D))
        tmp[0, :, :] = tmp0
        for i in xrange(1, self.n_features):
            newtmp = np.einsum('imj,klj->ikml', w2[i, x[i], :, :, :],
                               np.conjugate(w2[i, x[i], :, :, :])).reshape((self.D*self.D, self.D*self.D))
            tmp[i, :, :] = np.dot(tmp[i-1, :, :], newtmp)

        tmp2[self.n_features-1, :, :] = tmp1
        for i in xrange(self.n_features-2, -1, -1):
            newtmp = np.einsum('imj,klj->ikml', w2[i, x[i], :, :, :],
                               np.conjugate(w2[i, x[i], :, :, :])).reshape((self.D*self.D, self.D*self.D))
            tmp2[i, :, :] = np.dot(newtmp, tmp2[i+1, :, :])

        # Now for each tensor, the derivative is the contraction of the rest of the tensors

        # ~ temp1 = tmp[self.n_features-1, :,
            # ~ :].reshape(self.D*self.D, self.D, self.D)
        # ~ temp2 = tmp2[1, :, :].reshape(self.D, self.D, self.D*self.D)
        derivative[0, x[0], :, :, :] = 2*np.trace(np.einsum('ikm,ilrs->klrms',
                                                            w2[0, x[0], :, :, :], tmp2[1, :, :].reshape(self.D, self.D, self.D, self.D)), axis1=0, axis2=4)
        # ~ print(derivative[0,x[0],:,:,:].shape)
        # ~ temp1 = tmp[self.n_features-2, :,
        # ~ :].reshape(self.D*self.D, self.D, self.D)
        # ~ temp2 = tmp2[0, :, :].reshape(self.D, self.D, self.D*self.D)
        derivative[self.n_features-1, x[self.n_features-1], :, :, :] =\
            2*np.trace(np.einsum('ikm,irsj->krsmj', w2[self.n_features-1, x[self.n_features-1], :, :, :],
                                 tmp[self.n_features-2, :, :].reshape(self.D, self.D, self.D, self.D)), axis1=0, axis2=4)
        for i in xrange(1, self.n_features-1):
            temp1 = tmp[i-1, :, :].reshape(self.D*self.D, self.D, self.D)
            temp2 = tmp2[i+1, :, :].reshape(self.D, self.D, self.D*self.D)
            derivative[i, x[i], :, :, :] = 2*np.trace(np.einsum(
                'ikm,rij,kls->rjlms', w2[i, x[i], :, :, :], temp1, temp2), axis1=0, axis2=4)

        return derivative.reshape(self.m_parameters)

    def _derivativenorm(self):
        """Compute the derivative of the norm
        Returns
        -------
        derivative : numpy array, shape (m_parameters,)
        """

        w2 = np.reshape(
            self.w, (self.n_features, self.d, self.D, self.D, self.mu))
        derivative = np.zeros(
            (self.n_features, self.d, self.D, self.D, self.mu), dtype=np.complex128)

        tmp = np.zeros((self.n_features, self.D*self.D,
                        self.D*self.D), dtype=np.complex128)
        tmp2 = np.zeros((self.n_features, self.D*self.D,
                         self.D*self.D), dtype=np.complex128)

        tmp0 = np.einsum('pimj,pklj->ikml', w2[0, :, :, :, :],
                         np.conjugate(w2[0, :, :, :, :])).reshape((self.D*self.D, self.D*self.D))
        tmp1 = np.einsum('pimj,pklj->ikml', w2[self.n_features-1, :, :, :, :],
                         np.conjugate(w2[self.n_features-1, :, :, :, :])).reshape((self.D*self.D, self.D*self.D))
        tmp[0, :, :] = tmp0
        for i in xrange(1, self.n_features):
            newtmp = np.einsum('pimj,pklj->ikml', w2[i, :, :, :, :],
                               np.conjugate(w2[i, :, :, :, :])).reshape((self.D*self.D, self.D*self.D))
            tmp[i, :, :] = np.dot(tmp[i-1, :, :], newtmp)

        tmp2[self.n_features-1, :, :] = tmp1
        for i in xrange(self.n_features-2, -1, -1):
            newtmp = np.einsum('pimj,pklj->ikml', w2[i, :, :, :, :],
                               np.conjugate(w2[i, :, :, :, :])).reshape((self.D*self.D, self.D*self.D))
            tmp2[i, :, :] = np.dot(newtmp, tmp2[i+1, :, :])

        for j in xrange(self.d):
            # ~ temp1 = tmp[self.n_features-1, :,
            # ~ :].reshape(self.D*self.D, self.D, self.D)
            # ~ temp2 = tmp2[1, :, :].reshape(self.D, self.D, self.D*self.D)
            derivative[0, j, :, :, :] = 2*np.trace(np.einsum('ikm,ilrs->klrms',
                                                             w2[0, j, :, :, :], tmp2[1, :, :].reshape(self.D, self.D, self.D, self.D)), axis1=0, axis2=4)
            # ~ temp1 = tmp[self.n_features-2, :,
            # ~ :].reshape(self.D*self.D, self.D, self.D)
            # ~ temp2 = tmp2[0, :, :].reshape(self.D, self.D, self.D*self.D)
            derivative[self.n_features-1, j, :, :, :] = \
                2*np.trace(np.einsum('ikm,irsj->krsmj', w2[self.n_features-1, j, :, :, :],
                                     tmp[self.n_features-2, :, :].reshape(self.D, self.D, self.D, self.D)), axis1=0, axis2=4)
        for i in xrange(1, self.n_features-1):
            temp1 = tmp[i-1, :, :].reshape(self.D*self.D, self.D, self.D)
            temp2 = tmp2[i+1, :, :].reshape(self.D, self.D, self.D*self.D)
            for j in xrange(self.d):
                derivative[i, j, :, :, :] = 2*np.trace(np.einsum(
                    'ikm,rij,kls->rjlms', w2[i, j, :, :, :], temp1, temp2), axis1=0, axis2=4)

        return derivative.reshape(self.m_parameters)

    def _weightinitialization(self, rng):
        """Initialize weights w randomly
        Parameters
        ----------
        rng : random number generation
        """
        self.m_parameters = self.n_features*self.d*self.D*self.D*self.mu
        self.w = np.asarray(rng.normal(0, 1, self.m_parameters))\
            + 1j*np.asarray(rng.normal(0, 1, self.m_parameters))

    def _weightinitialization2(self, rng):
        """Initialize weights w randomly
        Parameters
        ----------
        rng : random number generation
        """
        self.m_parameters2 = (self.n_features)*self.d*self.D*self.D*self.mu
        return np.asarray(rng.normal(0, 1, self.m_parameters2))\
            + 1j*np.asarray(rng.normal(0, 1, self.m_parameters2))

    def _epsilon_padding(self, w, epsilon=0.0):
        """Reshaping function to add to the input parameters the unused parameters
        at the boundary conditions.
        Parameters
        ----------
        w : numpy array, shape (m_parameters2,)
        Returns
        -------
        w : numpy array, shape (m_parameters,)
        """
        # ~ num_samples = 2*len(self.w)
        noise = np.random.normal(0, epsilon, size=(
            self.n_features*self.d*self.D*self.D*self.mu, 2)).view(np.complex128)
        # ~ self.m_parameters2 = (self.n_features-2)*self.d*self.D*self.D*self.mu+self.d*self.D*self.mu*2
        self.m_parameters = (self.n_features)*self.d*self.D*self.D*self.mu
        new_w = np.zeros((self.n_features, self.d, self.D, self.D, self.mu), dtype=w.dtype) + \
            noise.reshape((self.n_features, self.d, self.D, self.D, self.mu))
        new_w[0, :, 0, :, :] = w[0:self.D*self.d *
                                 self.mu].reshape(self.d, self.D, self.mu)
        new_w[1:self.n_features-1, :, :, :, :] = w[self.D*self.d*self.mu *
                                                   2:].reshape((self.n_features-2, self.d, self.D, self.D, self.mu))
        new_w[self.n_features-1, :, :, 0, :] = w[self.D*self.d *
                                                 self.mu:self.D*self.d*self.mu*2].reshape(self.d, self.D, self.mu)
        return new_w.reshape(self.m_parameters)

    def _padding_function(self, w):
        """Reshaping function to add to the input parameters the unused parameters
        at the boundary conditions.
        Parameters
        ----------
        w : numpy array, shape (m_parameters2,)
        Returns
        -------
        w : numpy array, shape (m_parameters,)
        """
        new_w = np.zeros((self.n_features, self.d, self.D,
                          self.D, self.mu), dtype=w.dtype)
        new_w[0, :, 0, :, :] = w[0:self.D*self.d *
                                 self.mu].reshape(self.d, self.D, self.mu)
        new_w[1:self.n_features-1, :, :, :, :] = w[self.D*self.d*self.mu *
                                                   2:].reshape((self.n_features-2, self.d, self.D, self.D, self.mu))
        new_w[self.n_features-1, :, :, 0, :] = w[self.D*self.d *
                                                 self.mu:self.D*self.d*self.mu*2].reshape(self.d, self.D, self.mu)
        return new_w.reshape(self.m_parameters)

    def _unpadding_function(self, w):
        """Reshaping function to remove the unused parameters of the boundary conditions.
        Parameters
        ----------
        w : numpy array, shape (m_parameters,)
        Returns
        -------
        w : numpy array, shape (m_parameters2,)
        """
        self.m_parameters2 = (self.n_features-2)*self.d * \
            self.D*self.D*self.mu+self.d*self.D*self.mu*2
        w = w.reshape((self.n_features, self.d, self.D, self.D, self.mu))
        new_w = np.zeros(self.m_parameters2, dtype=w.dtype)
        new_w[0:self.D*self.d*self.mu] = w[0, :,
                                           0, :, :].reshape(self.d*self.D*self.mu)
        new_w[self.D*self.d*self.mu:self.D*self.d*self.mu *
              2] = w[self.n_features-1, :, :, 0, :].reshape(self.d*self.D*self.mu)
        new_w[self.D*self.d*self.mu*2:] = w[1:self.n_features-1, :, :, :,
                                            :].reshape((self.n_features-2)*self.d*self.D*self.D*self.mu)
        return new_w

    def _likelihood_derivative(self, v):
        """Compute derivative of log-likelihood of configurations in v
        Parameters
        ----------
        v : numpy array, shape (n_samples,n_features)
            Configurations
        Returns
        -------
        update_w : numpy array, shape (m_parameters,)
            array of derivatives of the log-likelihood
        """
        update_w = np.zeros(self.m_parameters, dtype=np.complex128)
        for n in xrange(v.shape[0]):
            update_w -= self._logderivative(v[n, :])
        update_w += v.shape[0]*self._logderivativenorm()
        update_w /= v.shape[0]
        return update_w
