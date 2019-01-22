import volrpynn.nest as v
import pyNN.nest as pynn
import numpy as np
import pytest

@pytest.fixture(autouse=True)
def setup():
    pynn.setup()

def test_nest_dense_create():
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(p1, p2, v.ReLU())
    expected_weights = np.ones((12, 10))
    actual_weights = d.projection.get('weight', format='array')
    assert not np.allclose(actual_weights, expected_weights) # Should be normal distributed
    assert abs(actual_weights.sum()) <= 24

def test_nest_dense_shape():
    p1 = pynn.Population(12, pynn.SpikeSourcePoisson(rate = 10))
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(p1, p2, v.ReLU(), weights = 1)
    pynn.run(1000)
    d.store_spikes()
    assert d.input.shape == (12,)
    assert d.output.shape[0] == 10

def test_nest_dense_projection():
    p1 = pynn.Population(12, pynn.SpikeSourcePoisson(rate = 10))
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    p2.record('spikes')
    d = v.Dense(p1, p2, v.ReLU(), weights = 1)
    pynn.run(1000)
    spiketrains = p2.get_data().segments[-1].spiketrains
    assert len(spiketrains) == 10
    avg_len = np.array(list(map(len, spiketrains))).mean()
    # Should have equal activation
    for train in spiketrains:
        assert abs(len(train) - avg_len) <= 1

def test_nest_dense_reduced_weight_fire():
    p1 = pynn.Population(2, pynn.SpikeSourcePoisson(rate = 1))
    p2 = pynn.Population(1, pynn.IF_cond_exp())
    p2.record('spikes')
    d = v.Dense(p1, p2, v.ReLU(), weights = np.array([[1], [0]]))
    pynn.run(1000)
    spiketrains = p2.get_data().segments[-1].spiketrains
    assert len(spiketrains) == 1
    assert spiketrains[0].size > 0

def test_nest_dense_increased_weight_fire():
    p1 = pynn.Population(1, pynn.SpikeSourcePoisson(rate = 1))
    p2 = pynn.Population(1, pynn.IF_cond_exp())
    p2.record('spikes')
    d = v.Dense(p1, p2, v.ReLU(), weights = 2)
    pynn.run(1000)
    spiketrains = p2.get_data().segments[-1].spiketrains
    count1 = spiketrains[0].size
    pynn.reset()
    p1 = pynn.Population(1, pynn.SpikeSourcePoisson(rate = 1))
    p2 = pynn.Population(1, pynn.IF_cond_exp())
    p2.record('spikes')
    d = v.Dense(p1, p2, v.ReLU(), weights = 2)
    pynn.run(1000)
    spiketrains = p2.get_data().segments[-1].spiketrains
    count2 = spiketrains[0].size
    assert count2 >= count1 * 2 

def test_nest_dense_chain():
    p1 = pynn.Population(12, pynn.SpikeSourcePoisson(rate = 100))
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    p3 = pynn.Population(2, pynn.IF_cond_exp())
    p3.record('spikes')
    d1 = v.Dense(p1, p2, v.ReLU())
    d2 = v.Dense(p2, p3, v.ReLU())
    pynn.run(1000)
    assert len(p3.get_data().segments[-1].spiketrains) > 0

def test_nest_dense_restore():
    p1 = pynn.Population(12, pynn.IF_cond_exp())
    p2 = pynn.Population(10, pynn.IF_cond_exp())
    d = v.Dense(p1, p2, v.ReLU(), weights = 2)
    d.set_weights(-1)
    assert np.array_equal(d.projection.get('weight', format='array'),
             np.ones((12, 10)) * -1)
    d.projection.set(weight = 1) # Simulate reset()
    assert np.array_equal(d.projection.get('weight', format='array'),
            np.ones((12, 10)))
    d.restore_weights()
    assert np.array_equal(d.projection.get('weight', format='array'),
            np.ones((12, 10)) * -1)

def test_nest_dense_backprop():
    p1 = pynn.Population(4, pynn.IF_cond_exp())
    p2 = pynn.Population(2, pynn.IF_cond_exp())
    l = v.Dense(p1, p2, v.UnitActivation(), weights = 1, decoder = lambda x: x)
    old_weights = l.get_weights()
    l.input = np.ones((1, 4)) # Mock spikes
    errors = l.backward(np.array([[0, 1]]), lambda w, g, b, bg: (w - g, b - bg))
    expected_errors = np.ones((4,)) - 13
    assert np.allclose(errors, expected_errors)
    expected_weights = np.tile([1, -3], (4, 1))
    assert np.allclose(l.get_weights(), expected_weights)

def test_nest_dense_numerical_gradient():
    # Test idea from https://github.com/stephencwelch/Neural-Networks-Demystified/blob/master/partSix.py
    # Use simple power function
    f = lambda x: x**2
    fd = lambda x: 2 * x
    e = 1e-4

    weights1 = np.ones((2, 3)).ravel()
    weights2 = np.ones((3, 1)).ravel()

    p1 = pynn.Population(2, pynn.IF_cond_exp())
    p2 = pynn.Population(3, pynn.IF_cond_exp())
    p3 = pynn.Population(1, pynn.IF_cond_exp())
    l1 = v.Dense(p1, p2, v.Sigmoid(), decoder = lambda x: x)
    l2 = v.Dense(p2, p3, v.Sigmoid(), decoder = lambda x: x)
    m = v.Model(l1, l2)
    error = v.SumSquared()

    def forward_pass(xs):
        "Simple sigmoid forward pass function"
        l1.input = xs
        l1.output = l2.input = v.Sigmoid()(np.matmul(xs, l1.weights))
        l2.output = v.Sigmoid()(np.matmul(l2.input, l2.weights))
        return l2.output

    def compute_numerical_gradient(xs, ys):
        "Computes the numerical gradient of a layer"
        weights1 = l1.get_weights().ravel() # 1D
        weights2 = l2.get_weights().ravel()
        weights = np.concatenate((weights1, weights2))
        gradients = np.zeros(weights.shape)

        def initialise_with_distortion(index, delta):
            distortion = np.copy(weights)
            distortion[index] = distortion[index] + delta
            l1.set_weights(distortion[:len(weights1)].reshape(l1.weights.shape))
            l2.set_weights(distortion[len(weights1):].reshape(l2.weights.shape))
            forward_pass(xs)

        # Calculate gradients
        for index in range(len(weights)):
            initialise_with_distortion(index, e)
            error1 = -error(l2.output, ys)
            initialise_with_distortion(index, -e)
            error2 = -error(l2.output, ys)
            gradients[index] = (error2 - error1) / (2 * e)
        
        # Reset weights
        l1.set_weights(weights1.reshape(2, 3))
        l2.set_weights(weights2.reshape(3, 1))

        return gradients

    def compute_gradients(xs, ys):
        class GradientOptimiser():
            counter = 2
            gradients1 = None
            gradients2 = None
            def __call__(self, w, wg, b, bg):
                if self.counter > 1:
                    self.gradients2 = wg
                else: 
                    self.gradients1 = wg
                self.counter -= 1
                return (w, b)
        output = forward_pass(xs)
        optimiser = GradientOptimiser()
        m.backward(error.prime(l2.output, ys), optimiser)
        return np.concatenate((optimiser.gradients1.ravel(), optimiser.gradients2.ravel()))

    # Normalise inputs
    xs = np.array(([3,5], [5,1], [10,2]), dtype=float)
    xs = xs - np.amax(xs, axis=0)
    ys = np.array(([75], [82], [93]), dtype=float)
    ys = ys / 100

    # Calculate numerical gradients
    numerical_gradients = compute_numerical_gradient(xs, ys)
    # Calculate 'normal' gradients
    gradients = compute_gradients(xs, ys)
    # Calculate the ratio between the difference and the sum of vector norms
    ratio = np.linalg.norm(gradients - numerical_gradients) /\
               np.linalg.norm(gradients + numerical_gradients)
    assert ratio < 1e-07
     
