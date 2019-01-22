import "layer_type"
import "../nn_types"
import "../util"
import "../weight_init"
import "../../../diku-dk/linalg/linalg"


-- | Merges an array of layers
module merge (R:real) : layer_type with t = R.t
                                   with input_params = (i32, i32)
                                   with activations  = activation_func ([]R.t)
                                   with input        = tup2d R.t
                                   with weights      = ()
                                   with output       = arr2d R.t
                                   with cache        = ()
                                   with error_in     = arr2d R.t
                                   with error_out    = tup2d R.t = {

  type t            = R.t
  type input        = tup2d t
  type weights      = ()
  type output       = arr2d t
  type cache        = ()
  type error_in     = arr2d t
  type error_out    = tup2d t
  type b_output     = (error_out, weights)

  type input_params = (i32, i32)
  type activations  = activation_func ([]t)

  type merge_tp = NN input weights output
                     cache error_in error_out (apply_grad t)

  module lalg   = mk_linalg R
  module util   = utility R
  module w_init = weight_initializer R

  -- Forward propagation
  let forward  (_:[]t -> []t)
               (_:bool)
               (_: weights)
               ((i1, i2):input) : (cache, output) =
    ((), map2 concat i1 i2)

  -- Backward propagation
  let backward (_:[]t -> []t) (l1_sz:i32)
               (_:bool)
               (_:apply_grad t)
               (_:weights)
               (_:cache)
               (error_concat:error_in) : b_output =
    (unzip (map (split l1_sz) error_concat), ())

  let init ((l, _):input_params) (act:activations) (_:i32) : merge_tp =
    {forward  = forward act.f,
     backward = backward act.fd l,
     weights  = ()}

}
