import numpy as np
import volrpynn.nest as v
import pyNN.nest as pynn



p1 = pynn.Population(2, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
p3 = pynn.Population(4, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
p5 = pynn.Population(2, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
layer0 = v.Dense(p1, p3, weights=np.random.normal(1.0, 1.0, (2, 4)), biases=0.0)
layer1 = v.Dense(p3, p5, weights=np.random.normal(1.0, 1.0, (4, 2)), biases=0.0)
l_decode = v.Decode(p5)
model = v.Model(layer0, layer1, l_decode)

optimiser = v.GradientDescentOptimiser(0.1, simulation_time=50.0)
if __name__ == "__main__":
    v.Main(model, 'parameters_xor_seq.npy').train(optimiser)

