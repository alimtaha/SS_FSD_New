
def lr_decay(global_step,
             init_learning_rate=1e-4,
             min_learning_rate=1e-7,
             decay_rate=0.999):


      print(type(init_learning_rate), type(min_learning_rate))
      lr = ((init_learning_rate - min_learning_rate) *
          pow(decay_rate, global_step // 10) +
          min_learning_rate)

      return lr
