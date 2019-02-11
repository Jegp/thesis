import numpy as np
import volrpynn.nest as v
import pyNN.nest as pynn



p1 = pynn.Population(100, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
p11 = pynn.Population(10, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
p13 = pynn.Population(20, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
p3 = pynn.Population(20, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
p5 = pynn.Population(20, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
p7 = pynn.Population(10, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
p9 = pynn.Population(20, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
p15 = pynn.Population(10, pynn.IF_cond_exp(**{"tau_syn_I":5,"tau_refrac":0,"v_thresh":-50,"v_rest":-65,"tau_syn_E":5,"v_reset":-65,"tau_m":20,"e_rev_I":-70,"i_offset":0,"cm":1,"e_rev_E":0}))
layer0 = v.Dense(p1, p3, weights=np.random.normal(1.0, 1.0, (100, 20)), biases=0.0)
layer3 = v.Replicate(p3, (p5, p9), weights=(np.random.normal(1.0, 1.0, (20, 20)), np.random.normal(1.0, 1.0, (20, 20))), biases=0.0)
layer1 = v.Dense(p5, p7, weights=np.random.normal(1.0, 1.0, (20, 10)), biases=0.0)
layer2 = v.Dense(p9, p11, weights=np.random.normal(1.0, 1.0, (20, 10)), biases=0.0)
layer4 = v.Merge((p7, p11), p13)
layer5 = v.Dense(p13, p15, weights=np.random.normal(1.0, 1.0, (20, 10)), biases=0.0)
l_decode = v.Decode(p15)
model = v.Model(layer0, layer1, layer2, layer3, layer4, layer5, l_decode)

optimiser = v.GradientDescentOptimiser(0.1, simulation_time=50.0)
if __name__ == "__main__":
    v.Main(model).train(optimiser)

