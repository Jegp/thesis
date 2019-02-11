import "../lib/github.com/HnimNart/deeplearning/deep_learning"
module dl = deep_learning f64
let x0 = dl.layers.dense (2, 4) dl.nn.relu 1
let x1 = dl.layers.dense (4, 2) dl.nn.relu 2
let x2 = dl.nn.connect_layers x0 x1

let nn = x2
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

