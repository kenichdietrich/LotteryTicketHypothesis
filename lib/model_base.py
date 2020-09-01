import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from . import pruning

from typing import Tuple, List
import pickle
import os
from collections import OrderedDict

""" Base classes for building models and sessions to perform
LTH experiments.
"""

class LotteryModel():
    """ Class to perform all steps in Lottery Ticket Hypothesis
    from a keras model built with custom Lottery layers found
    in layers.py. A model allows to perform a single experiment.

    Parameters
    ----------
        model (tf.keras.Model): Keras sequential model built with 
        custom layers.
        data (tuple): Tuple of data arrays. It can be length 2, 
        (x_train, y_train), or 4, (x_train, y_train, x_validation, 
        y_validation).
        optimizer (str): Keras name of optimizer used to train model.
        loss (str): Keras name of loss used to optimize the model.
        metrics (list): List of metrics used to evaluate the model.
        experiment (str): Experiment name.
    """
    def __init__(
        self,
        model: tf.keras.Model,
        data: Tuple[np.array],
        optimizer: str,
        loss: str,
        metrics: List[str],
        experiment: str = "DefaultExp"
    ):
        self.network = model
        self.data = data
        self.model_specs = {"optimizer": optimizer, 
                            "loss": loss, 
                            "metrics_list": metrics}
        self.experiment = experiment
        # Compile and build model using keras framework
        self.network.compile(optimizer=self.model_specs["optimizer"], 
                             loss=self.model_specs["loss"], 
                             metrics=self.model_specs["metrics_list"])
        self.network.build(input_shape = self.data[0].shape)
        # Store initial weights, important step in LTH
        self.initial_weights = pruning.save_weights(self.network)
        # Attribute to track the number of trainings
        self._n_training = 0
        # Attributes to track the number and state of prunings
        self._n_pruning = 0
        self._last_prop = 0
        # Attributes to track pruning criterion and reset mode
        self._last_criterion = None
        self._last_reinit = None
        # Attributes to save training histories and parameters
        self.history = OrderedDict()
        self.train_parameters = OrderedDict()

    def train(self, 
              epochs: int = 10,
              batch_size: int = 64,
              verbose: int = 1,
              is_one_shot: bool = False,
              is_iterative: bool = False
    ):
        """ Train the model and save results.

        Parameters
        ----------
            epochs (int): Number of epochs in training.
            batch_size (int): Number of samples in minibatch.
            verbose (int): It controlls training verbosity.
            is_one_shot (bool): True for one-shot experiments.
            is_iterative (bool): True for iterative experiments.
        """

        self._n_training += 1
        # Print output format
        title = "Full network" if self._n_pruning == 0 else str(round(self._last_prop*100))+"% Pruned network"
        print("{}: Training {}".format(self.experiment, title))
        # Training
        hist = self.network.fit(self.data[0], 
                                self.data[1], 
                                epochs=epochs, 
                                batch_size=batch_size, 
                                validation_data=self.data[2:],
                                verbose=verbose)
        # Select training name
        if self._n_pruning == 0:
            train_name = "Full network"

        else:
            if is_one_shot:
                train_name = "One-Shot "+str(round(self._last_prop*100))+"% pruned network"

            elif is_iterative:
                train_name = "Iterative "+str(round(self._last_prop*100))+"% pruned network"

            else:
                train_name = str(round(self._last_prop*100))+"% pruned network"

        # Store results
        self.history[train_name] = hist.history
        self.train_parameters[train_name] = {"epochs": epochs, 
                                             "batch_size": batch_size,
                                             "prop_pruned": self._last_prop,
                                             "pruning_criterion": self._last_criterion,
                                             "reinit": self._last_reinit}
    
    def prune(self, 
              prop: float = 0.5,
              pruning_criterion: str = "lf",
              is_iterative: bool = False,
              steps: int = 5,
              k: int = 0
    ):
        """ Prune weights by updating masks.
        
        Parameters
        ----------
            prop (float): Proportion of weights to prune in each layer. The last
            layer will be pruned at prop/2.
            pruning_criterion (str): Available "lf" (large final, default), 
            "lilf" (large init, large final), "mi" (magnitude increase), 
            "mov" (movement) and "lfss" (large final, same sign).
            is_iterative (bool): True for performing iterative pruning.
            steps (int): Number of iterative steps in the pruning process when 
            performing iterative pruning.
            k (int): Current step in the iterative process.
        """

        self._n_pruning += 1
        self._last_criterion = pruning_criterion

        if not is_iterative:
            self._last_prop = prop
            self.network = pruning.prune(model=self.network, 
                                         prop=prop, 
                                         initial_weights=self.initial_weights, 
                                         criterion=pruning_criterion)
        else:
            self._last_prop = round((k+1)*prop/steps, 2)
            self.network = pruning.iteration_pruning(model=self.network, 
                                                     k=k, 
                                                     it_prop=prop/steps,
                                                     initial_weights=self.initial_weights,
                                                     criterion=pruning_criterion)

    def reinit(self, 
               reset_mode: str = "rewind",
               same_sign: bool = True,
               jitter_sd: float = 0.01,
               constant: float = None
    ):

        """ Reinitialize weights in the model. 
        
        Parameters
        ----------
        reset_mode (str): How to reinitialise the unpruned weights. There are four available
        modes:
            "rewind": Rewind weights to the initial ones. Default mode.
            "jitter": Rewind and add noise to the initial weights.
            "random": Reinitialise with random weights based on the original init
            distribution.
            "reshuffle": Reinitialise by reshuffling the kept weights' initial values.
            "constant": Set the kept weights to a positive or negative constant.
        same_sign (bool): Specify whether same sign as original initialization is kept
        when resetting weights.
        jitter_sd (float): Standard deviation of added noise in "jitter" mode.
        constant (float): Constant of reinitialization when selecting "constant" mode.
        By default, this constant is set to the standard deviation of the original 
        distribution in each layer.
        """

        self._last_reinit = {"reset_mode": reset_mode,
                            "same_sign": same_sign,
                            "jitter_sd": jitter_sd,
                            "constant": constant if constant else "init sd"}

        self.network = pruning.reset_weights(self.network, 
                                             self.initial_weights,
                                             reset_mode=reset_mode,
                                             same_sign=same_sign,
                                             jitter_sd=jitter_sd,
                                             constant=constant)

    def one_shot(self,
                 prop: float = 0.5,
                 pruning_criterion: str = "lf",
                 epochs: int = 10,
                 batch_size: int = 64,
                 verbose: int = 1,
                 reset_mode: str = "rewind",
                 same_sign: bool = True,
                 jitter_sd: float = 0.01,
                 constant: float = None
    ):
        """ Pipeline-method to perform one-shot pruning. This method can
        only be used on a new (non-handled) model.
        
        Parameters
        ----------
            prop (float): Proportion of weights to prune in each layer.
            pruning_criterion (str): Available "lf" (large final, default), 
            "lilf" (large init, large final), "mi" (magnitude increase), 
            "mov" (movement) and "lfss" (large final, same sign).
            epochs (int): Number of epochs in training.
            batch_size (int): Number of samples in minibatch.
            verbose (int): It controlls training verbosity.
            reset_mode (str): How to initialise the unpruned weights. There are four available
            modes:
                "rewind": Rewind weights to the initial ones. Default mode.
                "jitter": Rewind and add noise to the initial weights.
                "random": Reinitialise with random weights based on the original init
                distribution.
                "reshuffle": Reinitialise by reshuffling the kept weights' initial values.
                "constant": Set the kept weights to a positive or negative constant.
            same_sign (bool): Specify whether same sign as original initialization is kept
            when resetting weights.
            jitter_sd (float): Standard deviation of added noise in "jitter" mode.
            constant (float): Constant of reinitialization when selecting "constant" mode.
            By default, this constant is set to the standard deviation of the original 
            distribution in each layer.
        """
        
        if self._n_training == 0 and self._n_pruning == 0:
            print("{}\nOne-Shot pruning\n{}".format("-"*72, "-"*72))
            # Train full network
            self.train(epochs=epochs, 
                       batch_size=batch_size, 
                       verbose=verbose, 
                       is_one_shot = True)
            # Pruning + reinitialisation
            self.prune(prop=prop, pruning_criterion=pruning_criterion)
            self.reinit(reset_mode=reset_mode, 
                        same_sign=same_sign, 
                        jitter_sd=jitter_sd, 
                        constant=constant)
            # Train pruned network
            self.train(epochs=epochs, 
                       batch_size=batch_size, 
                       verbose=verbose, 
                       is_one_shot = True)
    
        else:
            raise Exception("The model has already been handled. You"+
                            " must initialize a new model to perform"+
                            " this method.")

    def iterative_pruning(self,
                          prop: float = 0.5,
                          pruning_criterion: str = "lf",
                          steps: int = 5,
                          epochs: int = 10,
                          batch_size: int = 64,
                          verbose: int = 0,
                          reset_mode: str = "rewind",
                          same_sign: bool = True,
                          jitter_sd: float = 0.01,
                          constant: float = None
    ):
        """ Pipeline-method to perform iterative pruning. This algorithm 
        usually gives better results, although it requires a higher 
        computational cost. This method can only be used on a new 
        (non-handled) model.
        
        Parameters
        ----------
            prop (float): Proportion of weights to prune in each layer.
            pruning_criterion (str): Available "lf" (large final, default), 
            "lilf" (large init, large final), "mi" (magnitude increase), 
            "mov" (movement) and "lfss" (large final, same sign).
            steps (int): Number of iterative steps in the pruning process.
            The algorithm will prune a proportion of prop/steps at each step.
            epochs (int): Number of epochs in training.
            batch_size (int): Number of samples in minibatch.
            verbose (int): It controlls training verbosity.
            reset_mode (str): How to initialise the unpruned weights. There are four available
            modes:
                "rewind": Rewind weights to the initial ones. Default mode.
                "jitter": Rewind and add noise to the initial weights.
                "random": Reinitialise with random weights based on the original init
                distribution.
                "reshuffle": Reinitialise by reshuffling the kept weights' initial values.
                "constant": Set the kept weights to a positive or negative constant.
            same_sign (bool): Specify whether same sign as original initialization is kept
            when resetting weights.
            jitter_sd (float): Standard deviation of added noise in "jitter" mode.
            constant (float): Constant of reinitialization when selecting "constant" mode.
            By default, this constant is set to the standard deviation of the original 
            distribution in each layer.
        """
        
        if self._n_training == 0 and self._n_pruning == 0:
            print("{}\nIterative pruning\n{}".format("-"*72, "-"*72))
            # Train full network
            self.train(epochs=epochs, 
                       batch_size=batch_size, 
                       verbose=verbose, 
                       is_iterative=True)
            # Iterative process = prune + reinit + train
            for k in range(steps):
                self.prune(prop=prop, 
                           pruning_criterion=pruning_criterion, 
                           is_iterative=True, 
                           steps=steps, 
                           k=k)
                self.reinit(reset_mode=reset_mode, 
                            same_sign=same_sign, 
                            jitter_sd=jitter_sd, 
                            constant=constant)
                self.train(epochs=epochs, 
                           batch_size=batch_size, 
                           verbose=verbose, 
                           is_iterative=True)

        else:
            raise Exception("The model has already been handled. You"+
                            " must initialize a new model to perform"+
                            " this method.")

    def plot_learning(self, 
                      metric: str = None
    ):
        """ Method for plotting the learning curves. 

        Parameters
        ----------
            metric (str): Metric from the metrics list to plot. It plots first
            metric in list by default (parameter set to None).
        """

        if not metric:
            metric = self.model_specs["metrics_list"][0]

        else:
            if metric not in self.model_specs["metrics_list"]:
                print("This metric is not in your predefined list! The first metric "+ 
                "in list will be plotted instead.")
                metric = self.model_specs["metrics_list"][0]

        if self._n_training > 0:
            # There are four plots depicting the training, 
            # two for training and test loss (ax1, ax2) and
            # two for training and test metric (ax3, ax4)
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            for train, history in self.history.items():
                ax1.plot(history["loss"], label=train)
                ax2.plot(history["val_loss"], label=train)
                ax3.plot(history[metric], label=train)
                ax4.plot(history["val_"+metric], label=train)
            # Labels adn titles
            ax1.set_ylabel("loss"), ax1.legend(), ax1.set_title("Training set")
            ax2.set_ylabel("loss"), ax2.legend(), ax2.set_title("Test set")
            ax3.set_ylabel(metric), ax3.legend(), ax3.set_title("Training set")
            ax4.set_ylabel(metric), ax4.legend(), ax4.set_title("Test set")

            plt.suptitle("Lottery Ticket learning curves", size=14, fontweight="bold")

            plt.show()
        
        else:
            raise Exception("You haven't trained the model yet!")

    def plot_weights(self,
                     layer: int = 0
    ):
        """ Visualize the weights with a 2D scatter plot of final weights vs 
        initial weights.

        Parameters
        ----------
            layer (int): Layer whose weights will be plotted. First layer (0)
            by default.
        """

        n_layers = int(len(self.network.variables)/3)

        if layer < 0:
            raise Exception("Invalid index")

        elif layer >= n_layers:
            raise Exception("Number of layer exceeded!")
        
        else:
            mask = self.network.variables[layer*3+2].numpy() 
            wf = self.network.variables[layer*3].numpy()
            wi = self.initial_weights[self.network.variables[layer*3].name]
            # Proportion of weights pruned in the layer
            prop_pruned = 1 - mask.sum()/mask.size

            fig, ax = plt.subplots()
            fig.set_size_inches(6,6)
            # We only plot weights unpruned weights (where mask != 0)
            ax.scatter(wi[mask != 0], wf[mask != 0], s=0.2, c="#1D5EC2", marker=".")
            # Label showing the proportion of weights pruned
            ax.text(0.66, 0.1, "{}% pruned".format(round(prop_pruned*100)),
                    transform=ax.transAxes, fontweight="bold",
                    fontsize=12, bbox = dict(boxstyle="round", fc="0.8", alpha=0.3))
            ax.set_xlabel("$w_i$"), ax.set_ylabel("$w_f$")
            # Adjust limits
            lim1, lim2 = ax.get_ylim()
            ax.set_xlim(lim1, lim2)
            # Plot guide lines (axes and y=x line)
            ax.axvline(0, -1, 1, c="k", alpha=0.2, ls="--", lw = 1, dashes=[5,4])
            ax.axhline(0, -1, 1, c="k", alpha=0.2, ls="--", lw = 1, dashes=[5,4])
            ax.plot(0.9*np.array([lim1,lim2]), 0.9*np.array([lim1,lim2]), 
                    c="k", alpha=0.2, ls="--", lw = 1, dashes=[5,4])

            ax.set_title("Unpruned weights in layer {}\n{}: {} weights in total".format(layer, 
                            os.path.split(self.network.variables[layer*3].name)[0],
                            mask.size), 
                         size=14, fontweight="bold")

            plt.show()

    def pruning_summary(self):
        """ Print summary of pruning state. """

        layer_names = [os.path.split(m.name)[0] for m in self.network.variables if "_mask" in m.name]
        # Number of weights by layer
        weight_sizes = np.array([m.numpy().size for m in self.network.variables if "_mask" in m.name])
        # Number of unpruned weights by layer
        unpruned_weights = np.array([m.numpy().sum() for m in self.network.variables if "_mask" in m.name])
        # Pruned rate by layer
        pruned_by_layer = 1 - unpruned_weights/weight_sizes
        # Total pruned rate of the network
        total_pruned = 1 - unpruned_weights.sum()/weight_sizes.sum()
        # Print output format
        print("{: <30} {: >14} {: >14}\n{}".format("Layer", "Weights", "%Pruned", "-"*60))
        for i in range(len(layer_names)):
            print("{: <30} {: >14} {: >14}\n".format(layer_names[i], weight_sizes[i], round(pruned_by_layer[i]*100,2)))
        print("{}\n{: <30} {: >14} {: >14}".format("-"*60, "Total",sum(weight_sizes), round(total_pruned*100,2)))

    def save_results(self, name: str, path: str = ""):
        """ Save experiment results in pickle format. 
        Two objects are saved: history in the first place, train
        parameters in the second.

        Parameters
        ----------
            name (str): Name of the file containing the saved results
            (No extension .pickle is required).
            path (str): Directory where results will be saved. Current
            root directory by default. Avoid ending it with /.
        """

        if not name:
            raise Exception("You didn't input a filename!")

        else:
            with open(str(path)+"/"+str(name)+".pickle", "wb") as file:
                p = pickle.Pickler(file)
                p.dump(self.history)
                p.dump(self.train_parameters)


class LotterySession():
    """ Class to perform all steps in Lottery Ticket Hypothesis
    from a keras model built with custom Lottery layers found
    in layers.py. A Lottery session allows different experiments 
    to be carried out on the base model. 

    Parameters
    ----------
        model (tf.keras.Model): Keras model built with custom layers.
        data (tuple): Tuple of data arrays. It can be length 2, 
        (xtr, ytr), or 4, (xtr, ytr, xtest, ytest).
        optimizer (str): Keras name of optimizer used to train model.
        loss (str): Keras name of loss used to optimize the model.
        metrics (list): List of metrics used to evaluate the model.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        data: Tuple[np.array],
        optimizer: str,
        loss: str,
        metrics: List[str]
    ):
        # The base model is a LotteryModel instance
        self.model = LotteryModel(model, data, optimizer, loss, metrics)
        # Current experiment name and empty experiment flag
        self.current_experiment = None
        self._is_empty_experiment = None
        # List to store experiment names
        self.experiment_list = []
        # Attributes to save training histories and parameters
        self.history = OrderedDict()
        self.train_parameters = OrderedDict()
        # Reset flag
        self._is_reset = False
    
    def new_experiment(self, name: str = None):
        """ Start up an experiment, i.e. a LotteryModel instance.

        Parameters
        ----------
            name (str): Name of the new experiment.
        """

        if not name:
            raise Exception("You haven't named your experiment! Please, do it when"+
                  " initializing the experiment.")
        elif name in self.experiment_list:
            raise Exception("There is already an experiment with this name. Please,"+
                  " select another name for the new experiment.")
        else:
            # Avoid reset at the beginning, because it leads to a conflict that 
            # concerns metrics
            if self.experiment_list and not self._is_reset:
                self.reset_network()

            self.current_experiment = name
            self._is_empty_experiment = True
            self.experiment_list.append(name)
            self.model.experiment = name
            # Create a new entry in history and parameters dict to store results
            self.history[name] = OrderedDict()
            self.train_parameters[name] = OrderedDict()

    def reset_network(self, random: bool = False):
        """ Method to reset the masks and set the weights to the initial 
        ones, i.e. restarting the model keeping the histories obtained 
        so far saved.

        Parameters
        ----------
            random (bool): True for random reset of the weights (new model).
        """

        if not self._is_reset:
            # Reset masks
            self.model.network = pruning.reset_masks(self.model.network)

            if not random:
                self.model.reinit()
            else:
                # Reset weights randomly
                self.model.reinit(reset_mode="random")
                # Save the new initial weights
                self.model.initial_weights = pruning.save_weights(self.model.network)

            # Reset model flags
            self.model._n_training = 0
            self.model._n_pruning = 0
            self.model._last_prop = 0
            self.model._last_criterion = None
            self.model._last_reinit = None
            # Clean model dicts
            self.model.history = OrderedDict()
            self.model.train_parameters = OrderedDict()
            # Reset flag
            self._is_reset = True

        else:
            raise Exception("The model is already reset!")

    def one_shot(self,
                 prop: float = 0.5,
                 pruning_criterion: str = "lf",
                 epochs: int = 10,
                 batch_size: int = 64,
                 verbose: int = 1,
                 reset_mode: str = "rewind",
                 same_sign: bool = True,
                 jitter_sd: float = 0.01,
                 constant: float = None
    ):
        """ Pipeline-method to perform one-shot pruning.
        
        Parameters
        ----------
            prop (float): Proportion of weights to prune in each layer.
            pruning_criterion (str): Available "lf" (default), 
            "lilf", "mi", "mov" and "lfss".
            epochs (int): Number of epochs in training.
            batch_size (int): Number of samples in minibatch.
            verbose (int): It controlls training verbosity.
            reset_mode (str): How to initialise the unpruned weights. 
            There are four available modes:
                "rewind": Rewind weights to the initial ones. Default mode.
                "jitter": Rewind and add noise to the initial weights.
                "random": Reinitialise with random weights based on the original init
                distribution.
                "reshuffle": Reinitialise by reshuffling the kept weights' initial values.
                "constant": Set the kept weights to a positive or negative constant.
            same_sign (bool): Specify whether same sign as original initialization is kept
            when resetting weights.
            jitter_sd (float): Standard deviation of added noise in "jitter" mode.
            constant (float): Constant of reinitialization when selecting "constant" mode.
            By default, this constant is set to the standard deviation of the original 
            distribution in each layer.
        """
        
        if self.current_experiment and self._is_empty_experiment:
            # Train the LotteryModel
            self.model.one_shot(prop=prop, pruning_criterion=pruning_criterion,
                                epochs=epochs, batch_size=batch_size, verbose=verbose,
                                reset_mode=reset_mode, same_sign=same_sign, 
                                jitter_sd=jitter_sd, constant=constant)
            # Save results of LotteryModel instance in LotterySession dicts
            self.history[self.current_experiment] = self.model.history
            self.train_parameters[self.current_experiment] = self.model.train_parameters
            # Now the experiment has been handled
            self._is_empty_experiment = False
            self._is_reset = False

        else:
            raise Exception("You must start up a new empty experiment.")

    def iterative_pruning(self,
                          prop: float = 0.5,
                          pruning_criterion: str = "lf",
                          steps: int = 5,
                          epochs: int = 10,
                          batch_size: int = 64,
                          verbose: int = 0,
                          reset_mode: str = "rewind",
                          same_sign: bool = True,
                          jitter_sd: float = 0.01,
                          constant: float = None
    ):
        """ Pipeline-method to perform iterative pruning. This algorithm 
        usually gives better results, although it requires a higher 
        computational cost. This method can only be used on a new 
        (non-handled) model.
        
        Parameters
        ----------
            prop (float): Proportion of weights to prune in each layer.
            pruning_criterion (str): Available "lf" (default), 
            "lilf", "mi", "mov" and "lfss".
            steps (int): Number of iterative steps in the pruning process.
            The algorithm will prune a proportion of prop/steps at each step.
            epochs (int): Number of epochs in training.
            batch_size (int): Number of samples in minibatch.
            verbose (int): It controlls training verbosity.
            reset_mode (str): How to initialise the unpruned weights. 
            There are four available modes:
                "rewind": Rewind weights to the initial ones. Default mode.
                "jitter": Rewind and add noise to the initial weights.
                "random": Reinitialise with random weights based on the original init
                distribution.
                "reshuffle": Reinitialise by reshuffling the kept weights' initial values.
                "constant": Set the kept weights to a positive or negative constant.
            same_sign (bool): Specify whether same sign as original initialization is kept
            when resetting weights.
            jitter_sd (float): Standard deviation of added noise in "jitter" mode.
            constant (float): Constant of reinitialization when selecting "constant" mode.
            By default, this constant is set to the standard deviation of the original 
            distribution in each layer.
        """
        
        if self.current_experiment and self._is_empty_experiment:
            # Train the LotteryModel
            self.model.iterative_pruning(prop=prop, pruning_criterion=pruning_criterion,
                                         steps=steps, epochs=epochs, batch_size=batch_size, 
                                         verbose=verbose, reset_mode=reset_mode, 
                                         same_sign=same_sign, jitter_sd=jitter_sd, 
                                         constant=constant)
            # Save results of LotteryModel instance in LotterySession dicts
            self.history[self.current_experiment] = self.model.history
            self.train_parameters[self.current_experiment] = self.model.train_parameters
            # Now the experiment has been handled
            self._is_empty_experiment = False
            self._is_reset = False

        else:
            raise Exception("You must start up a new empty experiment.")
    
    def plot_learning(self, 
                      metric: str = None,
                      experiments: List[str] = None
    ):
        """ Method for plotting the learning curves. 

        Parameters
        ----------
            metric (str): Metric from the metrics list to plot. It plots first
            metric in list by default (parameter set to None).
            experiments (list): List of experiments whose curves will be plotted.
            By default it plots the current experiment results.
        """

        if not experiments:
            try:
                experiments = [self.experiment_list[-1]]

            except Exception:
                print("No experiments have been started.")

        if not metric:
            metric = self.model.model_specs["metrics_list"][0]

        else:
            if metric not in self.model.model_specs["metrics_list"]:
                print("This metric is not in your predefined list! The first metric "+ 
                "in list will be plotted instead.")
                metric = self.model.model_specs["metrics_list"][0]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        for experiment in experiments:
            if self.history[experiment]:
                for train, history in self.history[experiment].items():
                    label = experiment+": "+train
                    ax1.plot(history["loss"], label=label)
                    ax2.plot(history["val_loss"], label=label)
                    ax3.plot(history[metric], label=label)
                    ax4.plot(history["val_"+metric], label=label)
            else:
                raise Exception(experiment+": You haven't trained the model yet!")

        ax1.set_ylabel("loss"), ax1.legend(), ax1.set_title("Training set")
        ax2.set_ylabel("loss"), ax2.legend(), ax2.set_title("Test set")
        ax3.set_ylabel(metric), ax3.legend(), ax3.set_title("Training set")
        ax4.set_ylabel(metric), ax4.legend(), ax4.set_title("Test set")

        plt.suptitle("Lottery Ticket learning curves", size=14, fontweight="bold")

        plt.show()

    def pruning_summary(self):
        """ Print summary of pruning state in current experiment. """

        if self.current_experiment:
            print(self.current_experiment)
            self.model.pruning_summary()

        else:
            raise Exception("No experiment is started!")

    def save_results(self, name: str = None, path: str = ""):
        """ Method to save experiment results in pickle format. 
        Two objects are saved: history in the first place, train
        parameters in the second.

        Parameters
        ----------
            name (str): Name of the file containing the saved results
            (No extension .pickle is required).
            path (str): Directory where results will be saved. Current
            root directory by default. Avoid ending it with /.
        """

        if not name:
            raise Exception("You didn't input a filename!")

        else:
            with open(str(path)+"/"+str(name)+".pickle", "wb") as file:
                p = pickle.Pickler(file)
                p.dump(self.history)
                p.dump(self.train_parameters)


class LotteryRobustExperiment():
    """ Class to perform statistically robust LTH experiments. 
    This is done by performing several trials of the same experiment
    changing only the initial weights.
    This class opens a LotterySession and conducts the different
    trials within it.

    Parameters
    ----------
        model (tf.keras.Model): Keras model built with custom layers.
        data (tuple): Tuple of data arrays. It can be length 2, 
        (xtr, ytr), or 4, (xtr, ytr, xtest, ytest).
        optimizer (str): Keras name of optimizer used to train model.
        loss (str): Keras name of loss used to optimize the model.
        metrics (list): List of metrics used to evaluate the model.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        data: Tuple[np.array],
        optimizer: str,
        loss: str,
        metrics: List[str]
    ):
        # Start a LotterySession to perform the robust experiment
        self.session = LotterySession(model=model, data=data, 
                                      optimizer=optimizer,
                                      loss=loss, metrics=metrics)

        self.trials = 0
        # Flag for tracking the trials
        self._current_trial = 0
        # Dicts to store training results and a summary of them
        self.history = OrderedDict()
        self.summary = OrderedDict()
        # Flag for summary
        self._is_summarized = None

    def train(self,
              trials: int = 5,
              epochs: int = 5,
              batch_size: int = 64,
              prop: float = 0.5,
              pruning_criterion: str = "lf",
              is_iterative: bool = False,
              steps: int = 5,
              reset_mode: str = "rewind",
              same_sign: bool = True,
              jitter_sd: float = 0.01,
              constant: float = None
    ):
        """ Method to train the models of the robust experiment.

        Parameters
        ----------
            trials (int): Number of trials to perform = sample size.
            epochs (int): Number of epochs in training.
            batch_size (int): Number of samples in minibatch.
            prop (float): Proportion of weights to prune in each layer.
            pruning_criterion (str): Available "lf" (default), 
            "lilf", "mi", "mov" and "lfss".
            is_iterative (bool): True for iterative pruning.
            steps (int): Number of steps in iterative pruning.
            reset_mode (str): How to initialise the unpruned weights. There are four available
            modes:
                "rewind": Rewind weights to the initial ones. Default mode.
                "jitter": Rewind and add noise to the initial weights.
                "random": Reinitialise with random weights based on the original init
                distribution.
                "reshuffle": Reinitialise by reshuffling the kept weights' initial values.
                "constant": Set the kept weights to a positive or negative constant.
            same_sign (bool): Specify whether same sign as original initialization is kept
            when resetting weights.
            jitter_sd (float): Standard deviation of added noise in "jitter" mode.
            constant (float): Constant of reinitialization when selecting "constant" mode.
            By default, this constant is set to the standard deviation of the original 
            distribution in each layer.
        """

        self.trials += trials
        # Save original training parameters to avoid adding new results
        # with different specs
        if not self.history:
            self.train_specs = OrderedDict()
            self.train_specs["epochs"] = epochs
            self.train_specs["batch_size"] = batch_size
            self.train_specs["prop"] = prop
            self.train_specs["is_iterative"] = is_iterative
            self.train_specs["steps"] = steps
        
        # Check that the parameters are kept
        original_specs = [spec for _, spec in self.train_specs.items()]
        if original_specs == [epochs, batch_size, prop, is_iterative, steps]:
            if not is_iterative:
                for trial in range(self._current_trial, self._current_trial+trials):
                    print("\n** TRIAL {} **\n".format(trial+1))
                    self._current_trial = trial
                    self.session.new_experiment("Trial "+str(trial+1))
                    # Train the trial
                    self.session.one_shot(prop=prop, pruning_criterion=pruning_criterion, 
                                          epochs=epochs, batch_size=batch_size, verbose=0,
                                          reset_mode=reset_mode, same_sign=same_sign, 
                                          jitter_sd=jitter_sd, constant=constant)
                    # Reset the network to new random initial weights
                    self.session.reset_network(random=True)
                
                print("{}\n** The training is over **\n{}".format("*"*26, "*"*26))
                self._current_trial += 1
                # Save/update results
                self.history = self.session.history
            
            else:
                for trial in range(self._current_trial, self._current_trial+trials):
                    print("\n** TRIAL {} **\n".format(trial+1))
                    self._current_trial = trial
                    self.session.new_experiment("Trial "+str(trial+1))
                    # Train the trial
                    self.session.iterative_pruning(prop, pruning_criterion, steps, epochs, batch_size, verbose=0,
                                                   reset_mode=reset_mode, same_sign=same_sign, jitter_sd=jitter_sd,
                                                   constant=constant)
                    # Reset the network to new random initial weights                               
                    self.session.reset_network(random=True)
                
                print("\n{}\n** The training is over **\n{}".format("*"*26, "*"*26))
                self._current_trial += 1
                # Save the results of full network and last step
                for key, value in self.session.history.items():
                    self.history[key] = {}
                    for ind in (0, -1):
                        self.history[key][list(value.keys())[ind]] = value[list(value.keys())[ind]]

        else:
            raise Exception("You can't retrain by changing training parameters!")

        # The results has not been summarized yet
        self._is_summarized = False

    def get_summary(self):
        """ Compute a statistical summary of the results. """
        
        # Extract model names ("Full network", etc)
        model_names = list(self.history["Trial 1"].keys())
        # Extract metric names (loss, val_loss, etc)
        metric_names = list(self.history["Trial 1"][model_names[0]].keys())
        # Loop for unpacking and reorganizing results and computing summary values
        for model_name in model_names:
            # List (len=trials) containing dicts with metrics of model_name
            metrics = [v[model_name] for v in self.history.values()]
            # Create an entry for the trained model
            self.summary[model_name] = OrderedDict()

            for metric_name in metric_names:
                # For each metric, it gives a matrix of dims (trials, epochs)
                matrix = np.array([m_dict[metric_name] for m_dict in metrics])
                # Create an entry for the metric
                self.summary[model_name][metric_name] = OrderedDict()
                # Compute and store statistical values
                self.summary[model_name][metric_name]["mean"] = matrix.mean(axis=0)
                self.summary[model_name][metric_name]["median"] = np.median(matrix, axis=0)
                self.summary[model_name][metric_name]["min"] = matrix.min(axis=0)
                self.summary[model_name][metric_name]["max"] = matrix.max(axis=0)
                self.summary[model_name][metric_name]["sd"] = matrix.std(axis=0)
        # Now the results has been summarized
        self._is_summarized = True
        

    def save_results(self, name: str = None, path: str = ""):
        """ Method to save robust experiment results in pickle format.
        Two objects are saved: history in the first place, summary in 
        the second.
        
        Parameters
        ----------
            name (str): Name of the file containing the saved results
            (No extension .pickle is required).
            path (str): Directory where results will be saved. Current
            root directory by default. Avoid ending it with /.
        """

        if not name:
            raise Exception("You didn't input a filename!")
        else:
            with open(str(path)+"/"+str(name)+".pickle", "wb") as file:
                p = pickle.Pickler(file)
                p.dump(self.history)
                p.dump(self.summary)

    def plot_learning(self,
                      metric: str = None
    ):
        """ Method to plot robust learning curves. 
        
        Parameters
        ----------
            metric (str): Metric whose results will be depicted in the plot.
            If it's not specified or the input is invalid, the first metric
            in list will be taken by default.
        """

        if not metric:
            metric = self.session.model.model_specs["metrics_list"][0]

        else:
            if metric not in self.session.model.model_specs["metrics_list"]:
                print("This metric is not in your predefined list! The first metric "+ 
                "in list will be plotted instead.")
                metric = self.session.model.model_specs["metrics_list"][0]

        if self.history:
            if not self._is_summarized:
                self.get_summary()

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

            # The plot displays mean curves and confidence intervals of 1sd
            for trained_model in self.summary.keys():
                ax1.plot(self.summary[trained_model]["loss"]["mean"], label=trained_model)
                ax1.fill_between(range(self.train_specs["epochs"]),
                                 self.summary[trained_model]["loss"]["mean"]-
                                 self.summary[trained_model]["loss"]["sd"],
                                 self.summary[trained_model]["loss"]["mean"]+
                                 self.summary[trained_model]["loss"]["sd"],
                                 alpha=0.3, linewidth=0)
                ax2.plot(self.summary[trained_model]["val_loss"]["mean"], label=trained_model)
                ax2.fill_between(range(self.train_specs["epochs"]),
                                 self.summary[trained_model]["val_loss"]["mean"]-
                                 self.summary[trained_model]["val_loss"]["sd"],
                                 self.summary[trained_model]["val_loss"]["mean"]+
                                 self.summary[trained_model]["val_loss"]["sd"],
                                 alpha=0.3, linewidth=0)
                ax3.plot(self.summary[trained_model][metric]["mean"], label=trained_model)
                ax3.fill_between(range(self.train_specs["epochs"]),
                                 self.summary[trained_model][metric]["mean"]-
                                 self.summary[trained_model][metric]["sd"],
                                 self.summary[trained_model][metric]["mean"]+
                                 self.summary[trained_model][metric]["sd"],
                                 alpha=0.3, linewidth=0)
                ax4.plot(self.summary[trained_model]["val_"+metric]["mean"], label=trained_model)
                ax4.fill_between(range(self.train_specs["epochs"]),
                                 self.summary[trained_model]["val_"+metric]["mean"]-
                                 self.summary[trained_model]["val_"+metric]["sd"],
                                 self.summary[trained_model]["val_"+metric]["mean"]+
                                 self.summary[trained_model]["val_"+metric]["sd"],
                                 alpha=0.3, linewidth=0)
            ax1.set_ylabel("loss"), ax1.legend(), ax1.set_title("Training set")
            ax2.set_ylabel("loss"), ax2.legend(), ax2.set_title("Test set")
            ax3.set_ylabel(metric), ax3.legend(), ax3.set_title("Training set")
            ax4.set_ylabel(metric), ax4.legend(), ax4.set_title("Test set")

            plt.suptitle("Lottery Ticket learning curves: Robust experiment", 
                         size=14, fontweight="bold")

            plt.show()
        
        else:
            raise Exception("You haven't trained the model yet!")
