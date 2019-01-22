"""
The layers of VolrPyNN which must all define a method for a backward-pass
through the layers (to update the layer weights), as well as getting, setting and
storing weights.
"""
import abc
import numpy as np
import volrpynn as v
from volrpynn.util import get_pynn as pynn

class Layer():
    """A neural network layer with a PyNN-backed neural network population and a backwards
       weight-update function, based on existing spikes"""

    def __init__(self, pop_in, pop_out, gradient_model, decoder=v.spike_softmax):
        """
        Initialises a densely connected layer between two populations output
        Args:
        pop_in -- The input population
        pop_out -- The output population
        gradient_model -- An ActivationFunction that calculates the neuron gradients
                          given the current spikes and errors from this layer
        weights -- Either a single number, an array of weights or a generator object.
                   Defaults all weights to a normal distribution with mean 1.0
                   and standard deviation of 0.2
        decoder -- A function that can code a list of SpikeTrains into a numeric
                   numpy array
        wta -- Winner-takes-all is a boolean flag to set if inhibitory
               connections should be installed to inhibit the entire population
               after the first spike was fired
        """
        self.pop_in = pop_in
        self.pop_out = pop_out
        self.output = None
        self.weights = None

        # Store gradient model
        if not isinstance(gradient_model, v.ActivationFunction):
            raise ValueError("gradient_model must be an activation function")

        self.gradient_model = gradient_model

        # Store decoder
        assert callable(decoder), "spike decoder must be a function"
        self.decoder = decoder

    @staticmethod
    def _is_tuple(data, name, allow_none=False):
        """Tests that the given data is a tuple. If not, raise an exception
        using 'name'"""
        if (not data and not allow_none) and \
           (not isinstance(data, (tuple, list)) or len(data) != 2):
            raise ValueError(name + " must be tuple of length two")

    @abc.abstractmethod
    def backward(self, error, optimiser):
        """Performs backwards optimisation based on the given error and
        activation derivative"""

    @abc.abstractmethod
    def get_output(self):
        """Returns a numpy array of the decoded output"""

    def get_weights(self):
        """Returns the weights as a matrix of size (input, output)"""
        return self.weights

    def restore_weights(self):
        """Restores the current weights of the layer"""
        self.set_weights(self.get_weights())
        return self.weights

    @abc.abstractmethod
    def set_weights(self, weights):
        """Sets the weights of the network layer"""

    @abc.abstractmethod
    def store_spikes(self):
        """Stores the spikes of the current run"""

class Decode(Layer):
    """A layer that only decodes the spike trains without any activation passes"""

    def __init__(self, pop_in, decoder=v.spike_softmax):
        super(Decode, self).__init__(pop_in, None, v.UnitActivation(), decoder)
        self.weights = np.ones(pop_in.size)

    def backward(self, error, optimiser):
        return error

    def get_output(self):
        return self.decoder(self.output)

    def set_weights(self, weights):
        return self.weights

    def store_spikes(self):
        self.output = np.array(self.pop_in.getSpikes().segments[-1].spiketrains)
        return self

class Dense(Layer):
    """A densely connected neural layer between two populations,
       creating a PyNN all-to-all connection (projection) between the
       populations."""

    def __init__(self, pop_in, pop_out, gradient_model=v.ReLU(), weights=None,
                 biases=0, decoder=v.spike_count_normalised, projection=None):
        """
        Initialises a densely connected layer between two populations
        output
        """
        super(Dense, self).__init__(pop_in, pop_out, gradient_model, decoder)

        self.input = None

        # Create a projection between the input and output populations
        if not projection:
            connector = pynn().AllToAllConnector(allow_self_connections=False)
            projection = pynn().Projection(pop_in, pop_out, connector)
        self.projection = projection

        # Prepare spike recordings
        self.pop_in.record('spikes')
        self.pop_out.record('spikes')

        # Assign given weights or default to a normal distribution
        if weights is not None:
            self.set_weights(weights)
        else:
            random_weights = np.random.normal(0, 0.5, (pop_in.size, pop_out.size))
            self.set_weights(random_weights)

        if isinstance(biases, np.ndarray):
            self.biases = biases
        elif biases:
            self.biases = np.repeat(biases, pop_out.size)
        else:
            self.biases = np.zeros(pop_out.size)

    def backward(self, error, optimiser):
        """Backward pass in the dense layer

        Args:
        error -- The error in the output from this layer as a numpy array
        optimiser -- The optimiser that calculates the new layer weights, given
                     the current weights and the gradient deltas

        Returns:
        A tuple of the cached spikes from the first (input) layer and the errors
        """
        assert callable(optimiser), "Optimiser must be callable"

        try:
            self.input
        except AttributeError:
            raise RuntimeError("No input data found. Please simulate the model" +
                               " before doing a backward pass")

        # Calculate activations for output layer
        input_decoded = self.decoder(self.input)
        output_activations = np.matmul(input_decoded, self.weights)

        # Calculate layer delta and weight optimisations
        output_gradients = self.gradient_model.prime(output_activations + self.biases)
        delta = np.multiply(error, output_gradients)

        # Ensure correct multiplication of data
        if len(input_decoded.shape) == 1:
            weights_delta = np.outer(input_decoded, delta)
        else:
            weights_delta = np.matmul(input_decoded.T, delta)

        # Calculate weight and bias optimisation and store
        (new_weights, new_biases) = optimiser(self.weights, weights_delta,
                                              self.biases, delta)

        # NEST cannot handle too large weight values, so this guard
        # ensures that the simulation keeps running, despite large weights
        new_weights[new_weights > 1000] = 1000
        new_weights[new_weights < -1000] = -1000

        self.set_weights(new_weights)
        self.biases = new_biases

        # Return errors changes in backwards layer
        backprop = np.matmul(delta, self.weights.T)
        return backprop

    def get_output(self):
        return self.decoder(self.output)

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.projection.set(weight=weights)
        self.weights = self.projection.get('weight', format='array')

    def store_spikes(self):
        segments_in = self.projection.pre.get_data('spikes').segments
        self.input = np.array(segments_in[-1].spiketrains)
        segments_out = self.projection.post.get_data('spikes').segments
        self.output = np.array(segments_out[-1].spiketrains)
        return self

class Merge(Layer):
    """A merge layer that takes a tuple of input layers and
    uniforms them into a single output population by connecting them densely.
    In practice this happens by creating two dense layers between the two
    input populations and the output population."""

    def __init__(self, pop_in, pop_out, gradient_model=v.ReLU(), weights=None,
                 decoder=v.spike_count_normalised):
        super(Merge, self).__init__(pop_in, pop_out, gradient_model, decoder)
        Layer._is_tuple(pop_in, "Input populations")

        if pop_in[0].size + pop_in[1].size != pop_out.size:
            raise ValueError("Population input sizes must equal population output size")

        if not weights:
            weights = (None, None)

        self.layer1 = v.Dense(pop_in[0], pop_out, gradient_model=gradient_model,
                              weights=weights[0], decoder=decoder)
        self.layer2 = v.Dense(pop_in[1], pop_out, gradient_model=gradient_model,
                              weights=weights[1], decoder=decoder)

    def backward(self, error, optimiser):
        l1_error = self.layer1.backward(error, optimiser)
        l2_error = self.layer2.backward(error, optimiser)
        return (l1_error, l2_error)

    def get_output(self):
        # Layer1 output == layer2 output, because the population is the same
        return self.layer1.get_output()

    def get_weights(self):
        return (self.layer1.get_weights(), self.layer2.get_weights())

    def set_weights(self, weights):
        Layer._is_tuple(weights, "Layer weights")
        self.layer1.set_weights(weights[0])
        self.layer2.set_weights(weights[1])

    def store_spikes(self):
        self.layer1.store_spikes()
        self.layer2.store_spikes()

class Replicate(Layer):
    """A replicate layer that takes a single population and copies the outputs
    to the two output populations. In practice this happens by creating two dense
    layers between the input population and the output populations."""

    def __init__(self, pop_in, pop_out, gradient_model=v.ReLU(), weights=None,
                 decoder=v.spike_count_normalised):
        super(Replicate, self).__init__(pop_in, pop_out, gradient_model, decoder)
        Layer._is_tuple(pop_out, "Output populations")
        Layer._is_tuple(weights, "Replicate layer weights", allow_none=True)

        if not pop_out[0].size == pop_out[1].size or \
                (pop_out[0].size != pop_in.size):
            raise ValueError("Output populations must be of the same size as input")

        connector = pynn().AllToAllConnector(allow_self_connections=False)
        projection1 = pynn().Projection(pop_in, pop_out[0], connector)
        projection2 = pynn().Projection(pop_in, pop_out[1], connector)

        self.layer1 = v.Dense(pop_in, self.pop_out[0],
                              gradient_model=gradient_model,
                              weights=1, # Reset later
                              decoder=decoder, projection=projection1)
        self.layer2 = v.Dense(pop_in, self.pop_out[1],
                              gradient_model=gradient_model,
                              weights=1, # Reset later
                              decoder=decoder, projection=projection2)

        # Assign given weights or default to a normal distribution
        if weights is not None:
            self.set_weights(weights)
        else:
            random_weights = np.random.normal(0, 0.5, (2, pop_in.size, pop_out[0].size))
            self.set_weights(random_weights)

    def backward(self, error, optimiser):
        Layer._is_tuple(error, "Backwards error in replicate layer")
        l1_error = self.layer1.backward(error[0], optimiser)
        l2_error = self.layer2.backward(error[1], optimiser)
        return (l1_error + l2_error) / 2 # Return the mean

    def get_output(self):
        l1_output = self.layer1.get_output()
        l2_output = self.layer2.get_output()
        return np.array([l1_output, l2_output])

    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights
        self.layer1.set_weights(weights[0])
        self.layer2.set_weights(weights[1])

    def store_spikes(self):
        self.layer1.store_spikes()
        self.layer2.store_spikes()
