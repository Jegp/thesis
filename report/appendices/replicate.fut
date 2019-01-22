import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"

-- | Split input into several layers
module replicate (R:real) : layer_type with t = R.t
                                   with input_params = (i32)
                                   with activations  = activation_func ([]R.t)
                                   with input        = arr2d R.t
                                   with weights      = (std_weights R.t, std_weights R.t)
                                   with output       = tup2d R.t
                                   with cache        = (tup2d R.t, tup2d R.t)
                                   with error_in     = tup2d R.t
                                   with error_out    = arr2d R.t = {

  type t            = R.t
  type input        = arr2d t
  type weights      = (std_weights t, std_weights t)
  type output       = tup2d t
  type cache	    = (tup2d t, tup2d t)
  type error_in     = tup2d t
  type error_out    = arr2d t
  type b_output     = (error_out, weights)

  type input_params = (i32)
  type activations  = activation_func ([]t)

  type replicate_nn = NN input weights output
                      cache error_in error_out 
		      ((std_weights t) -> (std_weights t) -> (std_weights t))

  module lalg   = mk_linalg R
  module util   = utility R
  module w_init = weight_initializer R

  let empty_cache : (arr2d t, arr2d t) = ([[]],[[]])
  let empty_error : error_out = [[]]

  -- Forward propagation
  let forward (act:[]t -> []t)
               (training:bool)
               ((t1, t2): weights)
              (input:input) : (cache, output) =
    let f ((w, b): std_weights t): (tup2d t, arr2d t) =
      let res      = lalg.matmul w (transpose input)
      let res_bias = transpose (map2 (\xr b' -> map (\x -> (R.(x + b'))) xr) res b)
      let res_act  = map (\x -> act x) (res_bias)
      let cache    = if training then (input, res_bias) else empty_cache
      in (cache, res_act)
    let (c1, r1) = f t1
    let (c2, r2) = f t2
    in ((c1, c2), (r1, r2))

  -- Backward propagation
  let backward (act: []t -> []t)
               (first_layer: bool)
               (apply_grads: apply_grad t)
               ((w1, w2): weights)
               ((c1, c2): cache)
               ((e1, e2): error_in) : b_output =
    let b ((w, b): std_weights t) ((input, inp_w_bias): tup2d t) 
          (error: arr2d t) : (arr2d t, std_weights t) =
      let deriv    = (map (\x -> act x) inp_w_bias)
      let delta    = transpose (util.hadamard_prod_2d error deriv)
      let w_grad   = lalg.matmul delta input
      let b_grad   = map (R.sum) delta
      let (w', b') : std_weights t = apply_grads (w,b) (w_grad, b_grad)

      --- Calc error to backprop to previous layer
      let error' : arr2d t =
	if first_layer then
	 empty_error
	else
	 transpose (lalg.matmul (transpose w) delta)
      in (error', (w', b'))
    
    let (error1, w1) = b w1 c1 e1
    let (error2, w2) = b w2 c2 e2

    let zero = R.from_fraction 0 1
    let fact = (R.from_fraction 1 2) 
    let average_sum_matrix [l][m][n] (tensor: [l][m][n]t) : arr2d t=
      util.scale_matrix (reduce util.add_matrix (replicate m (replicate n zero)) tensor) fact

    in (average_sum_matrix [error1, error2], (w1, w2))

  let init (m:input_params) (act:activations) (seed:i32) : replicate_nn =
    let w = w_init.gen_random_array_2d_xavier_uni (m,m) seed
    let b = map (\_ -> R.(i32 0)) (0..<m)
    in {forward  = forward act.f,
	backward = backward act.fd,
	weights  = ((w, b), (w, b))}

}
