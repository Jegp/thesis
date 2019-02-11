import "lib/github.com/HnimNart/deeplearning/deep_learning"
module dl = deep_learning f64
let x0 = dl.layers.dense (100, 20) dl.nn.relu 0
let x1 = dl.layers.replicate 20 dl.nn.relu 1
let x2 = dl.nn.connect_layers x0 x1
let x3 = dl.layers.dense (20, 10) dl.nn.relu 3
let x4 = dl.layers.dense (20, 10) dl.nn.relu 4
let x5 = dl.nn.connect_parallel x3 x4
let x6 = dl.nn.connect_layers x2 x5
let x7 = dl.layers.merge (10, 10) dl.nn.relu 2
let x8 = dl.nn.connect_layers x6 x7
let x9 = dl.layers.dense (20, 10) dl.nn.relu 9
let x10 = dl.nn.connect_layers x8 x9

let nn = x10
let main [m] (input:[m][]dl.t) (labels:[m][]dl.t) =
  let batch_size = 128
  let train_l = i32.f64 (f64.i32 m * 0.8)
  let train = train_l - (train_l %% batch_size)
  let validation_l = i32.f64 (f64.i32 m * 0.2)
  let validation = validation_l - (validation_l %% batch_size)
  let alpha = 0.1
  let nn' = dl.train.gradient_descent nn alpha
            input[:train] labels[:train]
            batch_size dl.loss.softmax_cross_entropy_with_logits
  let acc = dl.nn.accuracy nn' input[train:train+validation]
     labels[train:train+validation] dl.nn.softmax dl.nn.argmax
  in (acc, nn'.weights)

