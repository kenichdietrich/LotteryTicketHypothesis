# Lottery Ticket Hypothesis API

This repository contains an API to perform experiments in the context of the Lottery Ticket Hypothesis [1]. It is integrated in the Keras and Tensorflow (>=2.3.0) framework.

## The API

The LTH API consists of three base classes. Each of them must be fed with a Keras sequential model (neither compiled nor built) made up with custom LTH layers (found in layers.py file) which can be combined with original Keras layers such as Flatten, Dropout and BatchNormalization without any problem. The three base classes are, from low to high level, the following:

* **LotteryModel**: It allows performing a simple LTH experiment on a model.
* **LotterySession**: A session is a set of experiments conducted on the same model.
* **LotteryRobust**: This class allows a statistically robust experiment to be carried out by repeating the same experiment by changing initial conditions.

These classes follow a hierarchy. LotteryModel constitutes the low-level class which is used by LotterySession in each experiment contained in the session and, in turn, LotteryRobust use a session to conduct the trials of the robust experiment.

![hierarchy ticket](/img_md/lotteryticket.pdf=250x)

Multiple pruning and reinitialisation methods are implemented, many of them inspired by those used in [2]. We are able to prune weights using the following criteria:

* *Large final* [$|w_f|$]: Common pruning criterion used in LTH, as for instance in [1]. It keeps weights that have large magnitude after training.
* *Large init, large final* [$min(\alpha|w_f|,|w_i|)$]: This criterion tends to select weights that are large both before and after training. 
* *Magnitude increase* [$|w_f|-|w_i|$]: A criterion which selects the weights that most increase their magnitude with training.
* *Movement* [$|w_f-w_i|$]: The weights that are furthest away from their initial values are kept using this criterion.
* *Large final, same sign* [$max(0,w_f w_i/|w_i|)$]: According to [2], the sign of the weights plays an important role in the LTH, so the large final criterion is also implemented taking into account the conservation of sign.

On the other hand, the reinitialisation is also an important step in the LTH, so several modes are implemented to reinitialise weights after pruning:

* *Rewind*: Classical reinit mode used in [1]. It simply set the unpruned weights to their initial values.
* *Jitter*: This mode rewinds the weights and adds white noise to them.
* *Random*: It reinitialise with random weights based on the original init distribution.
* *Reshuffle*: This is a mode that reinitialises the net by reshuffling the initial values of the kept weights.
* *Constant*: Set all the unpruned weights to a constant with variable sign.

In addition, there is an option to conserve the initial sign when reinitialising the weights, taking into account the results obtained in [2].

Finally, it should be noted that some methods are also implemented in the classes to plot the learning process and weights.

## References

[1] J. Frankle, M. Carbin. The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. 2018. [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)

[2] H. Zhou, J. Lan, R. Liu, J. Yosinski. Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask. 2020. [arXiv:1905.01067](https://arxiv.org/abs/1905.01067v4)
