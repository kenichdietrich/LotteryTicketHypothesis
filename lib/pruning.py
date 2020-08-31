import numpy as np
import tensorflow as tf

import os
from typing import Dict

""" Functions related to the pruning process and weight reinitialization.
"""

def save_weights(model: tf.keras.Model) -> Dict[str, np.array]:
    """ Function to save model initial weights. 
    
    Parameters
    ----------
        model (tf.keras.Model): Untrained built model.

    Returns
    -------
        dict: A dictionary containing variable (weight tensor) names as keys
        and tensors (as numpy arrays) as values.
    """

    return {w.name:w.numpy() for w in model.variables}


def prune(model: tf.keras.Model, 
          prop: float,
          initial_weights: Dict[str, np.array],
          criterion: str = "lf",
) -> tf.keras.Model:
    """ Function to prune a certain proportion of weights in each layer of the model.

    Parameters
    ----------
        model (tf.keras.Model): Trained model to prune. It must be made up of custom LTH layers.
        prop (float): Proportion of weights to prune in each layer.
        initial_weights (dict): Dictionary containing initial weights.
        criterion (str): Mask criterion for pruning the weights. By default it's set to "lf",
        which means "large final". Other criteria are "lilf" (large init, large final), 
        "mi" (magnitude increase) and "mov" (movement).

    Returns
    -------
        tf.keras.Model: Keras model pruned.
    """

    weights = {w.name:w for w in model.variables}
    # Filter kernel weights (names)
    kernel_names = [w.name for w in model.variables if ("_bias" not in w.name) and ("_mask" not in w.name)]
    
    for w_name in kernel_names:
        wf = weights[w_name].numpy()
        wi = initial_weights[w_name]
        scores = pruning_criterion(wf, wi, prop, criterion)
        if w_name != kernel_names[-1]:
            quantile = np.quantile(scores, prop)
        else:
            # Prune the last layer at half of prop (as in the original paper).
            quantile = np.quantile(scores, prop/2)

        new_mask = scores > quantile
        weights[w_name+"_mask:0"].assign(new_mask)
    
    return model


def iteration_pruning(model: tf.keras.Model, 
                      k: int, 
                      it_prop: float,
                      initial_weights: Dict[str, np.array],
                      criterion: str = "lf"
) -> tf.keras.Model:
    """ Function to perform iterative pruning. This function represents a 
    single iteration, so it should usually be within a loop.

    Parameters
    ----------
        model (tf.keras.Model): Trained model to prune. It must be made up of custom LTH layers.
        k (int): Iterative pruning step.
        it_prop (float): Proportion of weights to prune at each step and in each layer.
        initial_weights (dict): Dictionary containing initial weights.
        criterion (str): Mask criterion for pruning the weights. By default it's set to "lf",
        which means "large final". Other criteria are "lilf" (large init, large final), 
        "mi" (magnitude increase) and "mov" (movement).

    Returns
    -------
        tf.keras.Model: Pruned Keras model at step k.
    """
    # Check there are weights to prune
    if k*it_prop < 1:
        weights = {w.name:w for w in model.variables}
        kernel_names = [w.name for w in model.variables if ("_bias" not in w.name) and ("_mask" not in w.name)]

        for w_name in kernel_names:
            wf = weights[w_name].numpy()*weights[w_name + "_mask:0"].numpy()
            wi = initial_weights[w_name]
            # Drop all zeros (weights already pruned)
            wfnon0 = wf[wf!=0]
            winon0 = wi[wf!=0]

            if w_name != kernel_names[-1]:
                # Depending on k, the quantile must be adapted to give a general
                # pruning prop percentage.
                # Scores of unpruned weights
                scores = pruning_criterion(wfnon0, winon0, it_prop/(1-k*it_prop), criterion)
                quantile = np.quantile(scores, it_prop/(1-k*it_prop))
            else:
                # Prune the last layer at half of prop (as in the original paper).
                scores = pruning_criterion(wfnon0, winon0, it_prop/(2-k*it_prop), criterion)
                quantile = np.quantile(scores, it_prop/(2-k*it_prop))
            # Compute the scores again in matrix form, dropping out the weights 
            # already pruned by adding -999
            new_mask = pruning_criterion(wf, 
                                         wi, 
                                         it_prop/(1-k*it_prop), 
                                         criterion)*weights[w_name + "_mask:0"].numpy() + np.abs(weights[w_name + "_mask:0"].numpy()-1)*(-999) > quantile

            weights[w_name+"_mask:0"].assign(new_mask)
    else:
        print("There are no weights to prune anymore!")
    
    return model


def pruning_criterion(wf: np.array,
                    wi: np.array,
                    prop: float,
                    criterion: str = "lf",
) -> np.array:
    """ Pruning criteria. This function returns a tensor with the same
    shape as the input giving the mask scores for the weights, specifying 
    the pruning criterion. The criteria available are:
    *Large final (lf):  M(wi, wf) = |wf|
    *Large init, large final (lilf):   M(wi, wf) = min(a*|wf|, |wi|)
    *Magnitude increase (mi):    M(wi, wf) = |wf| - |wi|
    *Movement (mov):    M(wi, wf) = |wf - wi|
    *Large final, same sign (lfss):    M(wi, wf) = max(0, wi*wf/|wi|)


    Parameters
    ----------
        wf (np.array): Tensor of trained weights to be pruned.
        wi (np.array): Initial weights.
        prop (float): Proportion (percentile) to prune.
        criterion (str): Mask criterion for computing the scores.

    Returns
    -------
        np.array: Tensor of mask scores assigned to each weight.
    """

    if criterion == "lf":
        return np.abs(wf)

    elif criterion == "lilf":
        qf = np.quantile(np.abs(wf), prop)
        qi = np.quantile(np.abs(wi), prop)
        # Scale parameter to align percentiles
        a = qi/qf

        return np.minimum(a*np.abs(wf), np.abs(wi))
    
    elif criterion == "mi":
        return np.abs(wf)-np.abs(wi)

    elif criterion == "mov":
        return np.abs(wf-wi)

    elif criterion == "lfss":
        return np.maximum(np.zeros(wi.shape,"float32"), wi*wf/np.abs(wi))

    else:
        raise Exception("No criterion with that name!")
        

def reset_weights(model: tf.keras.Model, 
                  initial_weights: Dict[str, np.array],
                  reset_mode: str = "rewind",
                  same_sign: bool = True,
                  jitter_sd: float = 0.01,
                  constant: float = None
) -> tf.keras.Model:
    """ Function to reinitialize weights (kernel+bias) of the pruned network in order
    to perform LTH.

    Parameters
    ----------
        model (tf.keras.Model): Pruned model to reinitialize.
        initial_weights (Dict[str, np.array]): Dictionary of initial weights saved with 
        save_weights function.
        reset_mode (str): How to initialise the unpruned weights. There are five available
        modes:
            "rewind": Rewind weights to the initial ones. Default mode.
            "jitter": Rewind and add noise to the initial weights.
            "random": Reinitialise with random weights based on the original init
            distribution.
            "reshuffle": Reinitialise by reshuffling the kept weights initial values.
            "constant": Set the kept weights to a positive or negative constant.
        same_sign (bool): Specify whether same sign as original initialization is kept
        when resetting weights.
        jitter_sd (float): Standard deviation of added noise in "jitter" mode.
        constant (float): Constant of reinitialization when selecting "constant" mode.
        By default, this constant is set to the standard deviation of the original 
        distribution in each layer.
    
    Returns
    -------
        tf.keras.Model: Keras pruned model reinitialized.
    """
    
    if reset_mode == "rewind":

        weights = {w.name:w for w in model.variables}
        # Filter kernel and bias weights
        kandb_names = [w.name for w in model.variables if ("_mask" not in w.name)]
        
        for w_name in kandb_names:
            w0 = initial_weights[w_name]
            weights[w_name].assign(w0)
        # Compile model to reset optimizer
        model.compile(optimizer = model.optimizer._name, 
                    loss = model.loss,
                    metrics=[m for m in model.metrics_names if m != "loss"]
                    )
    
        return model

    elif reset_mode == "jitter":

        return jitter_reset(model, initial_weights, jitter_sd)

    elif reset_mode == "random":

        weights = {w.name:w for w in model.variables}
        # Filter kernel weights
        kernel_names = [w.name for w in model.variables if ("_bias" not in w.name) and ("_mask" not in w.name)]
        # Dictionary containing initialization sd for each layer
        init_sd = {layer.name:layer.init_sd for layer in model.layers}
    
        for w_name in kernel_names:
            w0 = initial_weights[w_name]
            # Extract layer name
            l_name = os.path.split(w_name)[0]
            # Select custom standard deviation for layer
            sd = init_sd[l_name]
            k_init = tf.random_normal_initializer(stddev=sd)
            w = k_init(shape = weights[w_name].shape, dtype = "float32").numpy()
            if same_sign:
                # Due to the probabilistic symmetry, we can simply reassign the signs while
                # preserving the distribution.
                correct_signs = (w0 >= 0)*(w >= 0)+(w0 < 0)*(w < 0)
                signs_mask = correct_signs + (correct_signs-1)
                w = w*signs_mask

            weights[w_name].assign(w)
        # Filter bias weights
        bias_names = [w.name for w in model.variables if "_bias" in w.name]

        for w_name in bias_names:
            weights[w_name].assign(np.zeros(weights[w_name].shape, dtype="float32"))

        # Compile model to reset optimizer
        model.compile(optimizer = model.optimizer._name, 
                    loss = model.loss,
                    metrics=[m for m in model.metrics_names if m != "loss"]
                    )
        
        return model

    elif reset_mode == "reshuffle":

        weights = {w.name:w for w in model.variables}
        # Filter kernel weights
        kernel_names = [w.name for w in model.variables if ("_bias" not in w.name) and ("_mask" not in w.name)]
        
        for w_name in kernel_names:
            w0 = initial_weights[w_name].copy()
            mask = weights[w_name+"_mask:0"].numpy()
            sample = (w0*mask)[(w0*mask) != 0]
            np.random.shuffle(sample)
            inds = np.where(mask > 0.5)
            # Loop to assign shuffled weights
            if not same_sign:
                for n,index in enumerate(zip(*inds)):
                    w0[index] = sample[n]
            else:
                pos_sample = sample[sample >= 0]
                neg_sample = sample[sample < 0]
                pos_ind, neg_ind = 0, 0
                
                for index in zip(*inds):
                    if w0[index] >= 0:
                        w0[index] = pos_sample[pos_ind]
                        pos_ind += 1
                    else:
                        w0[index] = neg_sample[neg_ind]
                        neg_ind += 1

            weights[w_name].assign(w0)

        # Filter bias weights
        bias_names = [w.name for w in model.variables if "_bias" in w.name]

        for w_name in bias_names:
            weights[w_name].assign(np.zeros(weights[w_name].shape, dtype="float32"))

        # Compile model to reset optimizer
        model.compile(optimizer = model.optimizer._name, 
                    loss = model.loss,
                    metrics=[m for m in model.metrics_names if m != "loss"]
                    )
    
        return model

    elif reset_mode == "constant":

        weights = {w.name:w for w in model.variables}
        # Filter kernel weights
        kernel_names = [w.name for w in model.variables if ("_bias" not in w.name) and ("_mask" not in w.name)]
        
        init_sd = {layer.name:layer.init_sd for layer in model.layers}

        for w_name in kernel_names:
            l_name = os.path.split(w_name)[0]
            w0 = initial_weights[w_name]
            w = w0.copy()
            mask = weights[w_name+"_mask:0"].numpy()
            if not constant:
                constant = init_sd[l_name]
            inds = np.where(mask > 0.5)
            
            # Loop to assign shuffled weights
            for index in zip(*inds):
                w[index] = constant
            
            if same_sign:
                signs_mask = (w0 >= 0) + ((w0 >= 0)-1)
                w = w*signs_mask
            else:
                random_mask = np.random.rand(*mask.shape)-0.5 >= 0
                signs_mask = random_mask+(random_mask-1)
                w = w*signs_mask

            weights[w_name].assign(w)

        # Filter bias weights
        bias_names = [w.name for w in model.variables if "_bias" in w.name]

        for w_name in bias_names:
            weights[w_name].assign(np.zeros(weights[w_name].shape, dtype="float32"))

        # Compile model to reset optimizer
        model.compile(optimizer = model.optimizer._name, 
                    loss = model.loss,
                    metrics=[m for m in model.metrics_names if m != "loss"]
                    )
    
        return model

    else:
        raise Exception("Invalid mode!")

def random_reset_weights(model):
    """ Function to reset weights randomly.

    Parameters
    ----------
        model (tf.keras.Model): Pruned model.

    Returns
    -------
        tf.keras.Model: Keras pruned model randomly reinitialized.
    """

    weights = {w.name:w for w in model.variables}
    # Filter kernel weights
    kernel_names = [w.name for w in model.variables if ("_bias" not in w.name) and ("_mask" not in w.name)]
    # Dictionary containing initialization sd for each layer
    init_sd = {layer.name:layer.init_sd for layer in model.layers}
    
    for w_name in kernel_names:
        # Extract layer name
        l_name = os.path.split(w_name)[0]
        # Select custom standard deviation for layer
        sd = init_sd[l_name]
        k_init = tf.random_normal_initializer(stddev=sd)
        weights[w_name].assign(k_init(shape = weights[w_name].shape, 
                                   dtype = "float32").numpy()
                              )
    # Filter bias weights
    bias_names = [w.name for w in model.variables if "_bias" in w.name]

    for w_name in bias_names:
        weights[w_name].assign(np.zeros(weights[w_name].shape, dtype="float32"))

    # Compile model to reset optimizer
    model.compile(optimizer = model.optimizer._name, 
                  loss = model.loss,
                  metrics=[m for m in model.metrics_names if m != "loss"]
                 )
    
    return model


def jitter_reset(model: tf.keras.Model, 
                initial_weights: Dict[str, np.array], 
                sd: float = 0.01) -> tf.keras.Model:
    """ Function to add white noise to the weights.

    Parameters
    ----------
        model (tf.keras.Model): Pruned model.
        initial_weights(Dict[str, np.array]): Dictionary containing initial weights saved 
        with save_weights function.
        sd (float): Standard deviation of noise added.

    Returns
    -------
        tf.keras.Model: Jittered Keras model.
    """

    weights = {w.name:w for w in model.variables}
    # Filter kernel weights
    kernel_names = [w.name for w in model.variables if ("_bias" not in w.name) and ("_mask" not in w.name)]
    
    k_init = tf.random_normal_initializer(stddev = sd)
    for w_name in kernel_names:
        w0 = initial_weights[w_name]
        noise = k_init(shape = w0.shape, dtype="float32")
        weights[w_name].assign(w0+noise)
    # Filter bias weights
    bias_names = [w.name for w in model.variables if "_bias" in w.name]

    for w_name in bias_names:
        weights[w_name].assign(np.zeros(weights[w_name].shape, dtype="float32"))

    # Compile model to reset optimizer
    model.compile(optimizer = model.optimizer._name, 
                  loss = model.loss,
                  metrics=[m for m in model.metrics_names if m != "loss"]
                 )
    
    return model


def reset_masks(model: tf.keras.Model):
    """ Function to reset pruned masks. 
    
    Parameters
    ----------
        model (tf.keras.Model): Keras model whose masks will be reset.

    Returns
    -------
        tf.keras.Model: Model with reset masks (all elements set to 1.).
    """

    weights = {w.name:w for w in model.variables}
    # Filter masks
    mask_names = [w.name for w in model.variables if "_mask" in w.name]

    for m_name in mask_names:
        mask = weights[m_name]
        new_mask = np.ones(mask.shape, dtype="float32")
        mask.assign(new_mask)
    
    return model