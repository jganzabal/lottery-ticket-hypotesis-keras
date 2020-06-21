import tensorflow_model_optimization as tfmot
import numpy as np
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras import optimizers
import tensorflow as tf


def get_default_layers(model):
    layers_to_prune = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
            layers_to_prune.append(layer.name)
    return layers_to_prune
    
def initialize_pruned_model(pruned_model):
       
    pruned_model.compile(optimizer=optimizers.SGD(lr=0), loss='sparse_categorical_crossentropy', metrics='accuracy')
        
    X_train = np.random.normal(0, 0.2, [1] + pruned_model.input.shape.as_list()[1:])
    if pruned_model.loss == 'sparse_categorical_crossentropy':
        y_train = np.array([np.random.randint(pruned_model.output_shape[1])]).reshape(1, -1)
    else:
        y_train = np.random.normal(0, 0.2, pruned_model.output_shape[1]).reshape(1, -1)
        
    pruned_model.fit(X_train[0:1], y_train[0:1], epochs=1, verbose=0, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
    return pruned_model

def is_pruned(model):
    for layer in model.layers:
        if isinstance(layer, tfmot.sparsity.keras.pruning_wrapper.PruneLowMagnitude):
            return True
    return False

def prune_and_initilize(trained_model, pm, initial_weights, layers_to_prune=None):
    sparcity = 1 - pm
    sprasity_sched = tfmot.sparsity.keras.ConstantSparsity(
        sparcity, 
        0, # Do sparcity calculation in the first step
        end_step=0, # Do it only once
        frequency=10000000
    )
    if not is_pruned(trained_model):
        model = clone_model(trained_model)
        model.set_weights(trained_model.get_weights())
        
        if layers_to_prune is None:
            layers_to_prune = get_default_layers(model)

        prunned_model_layers = []
        for layer in model.layers:
            if layer.name in layers_to_prune:
                prunned_model_layers.append(tfmot.sparsity.keras.prune_low_magnitude(layer, sprasity_sched))
            else:
                prunned_model_layers.append(layer)

        trained_pruned_model = Sequential(prunned_model_layers)
        # Calculates mask
        initialize_pruned_model(trained_pruned_model)
    else:
        trained_pruned_model = trained_model
        model = tfmot.sparsity.keras.strip_pruning(trained_model)
        
    model.load_weights(initial_weights)
    prunned_model_layers = []
    for i, layer in enumerate(trained_pruned_model.layers):
        if isinstance(layer, tfmot.sparsity.keras.pruning_wrapper.PruneLowMagnitude):
            l_weights = model.layers[i].get_weights()
            l_weights[0] = l_weights[0]*layer.pruning_vars[0][1].numpy()
            model.layers[i].set_weights(l_weights)        
            prunned_model_layers.append(tfmot.sparsity.keras.prune_low_magnitude(model.layers[i], sprasity_sched))
        else:
            prunned_model_layers.append(model.layers[i])
    untrained_prunned_model = Sequential(prunned_model_layers)
    untrained_prunned_model.compile(optimizer=optimizers.SGD(lr=0), loss='sparse_categorical_crossentropy', metrics='accuracy')
    return untrained_prunned_model

def get_prunned_model(trained_model, pm=0.5, X_train=None, y_train=None, layers_to_prune=None):
    """
    Given a filename with weights, and a list with layers to prune, returns a pruned model with correct mask
    X_train, y_train are necesary to get the mask calculated by keras (Needs a fit) pm = 1 - sparcity as mentioned in paper
    """

    model = clone_model(trained_model)
    model.set_weights(trained_model.get_weights())

    sparcity = 1 - pm
    sprasity_sched = tfmot.sparsity.keras.ConstantSparsity(
        sparcity, 
        0, # Do sparcity calculation in the first step
        end_step=0, # Do it only once
        frequency=10000000
    )

    if layers_to_prune is None:
        layers_to_prune = get_default_layers(model)

    prunned_model_layers = []
    for layer in model.layers:
        if layer.name in layers_to_prune:
            prunned_model_layers.append(tfmot.sparsity.keras.prune_low_magnitude(layer, sprasity_sched))
        else:
            prunned_model_layers.append(layer)

    pruned_model = Sequential(prunned_model_layers)
    del model
    # This is necesary to make keras calculate the mask, learning rate is 0
    initialize_pruned_model(pruned_model)

    return pruned_model

def initialize_sparse_model(trained_model, pruned_model_with_mask, pm):
    """
        Given a filename (or a model) with weights and a pruned model with its mask, returns a new model with weights in filename and pruned with mask
    """
    model = clone_model(trained_model)
    model.set_weights(trained_model.get_weights())

    sparcity = 1 - pm
    sprasity_sched = tfmot.sparsity.keras.ConstantSparsity(
        sparcity, 
        0, # Do sparcity calculation in the first step
        end_step=0, 
        frequency=10000000
    )

    prunned_model_layers = []
    for i, layer in enumerate(pruned_model_with_mask.layers):
        if isinstance(layer, tfmot.sparsity.keras.pruning_wrapper.PruneLowMagnitude):
            l_weights = model.layers[i].get_weights()
            l_weights[0] = l_weights[0]*layer.pruning_vars[0][1].numpy()
            model.layers[i].set_weights(l_weights)        
            prunned_model_layers.append(tfmot.sparsity.keras.prune_low_magnitude(model.layers[i], sprasity_sched))
        else:
            prunned_model_layers.append(model.layers[i])
    prunned_model = Sequential(prunned_model_layers)
    prunned_model.compile(optimizer=optimizers.SGD(lr=0), loss='sparse_categorical_crossentropy', metrics='accuracy')
    return prunned_model
    
    
class LTH:
    def __init__(self, get_model):
        self.get_model = get_model
    
    def test(self, a):
        print(a)
    
    
    
    
        
    
    def get_prunned_model(self, model_or_file, pm=0.5, X_train=None, y_train=None, layers_to_prune=None):
        """
        Given a filename with weights, and a list with layers to prune, returns a pruned model with correct mask
        X_train, y_train are necesary to get the mask calculated by keras (Needs a fit) pm = 1 - sparcity as mentioned in paper
        """
        
        if type(model_or_file) == str:
            model = self.get_model()
            model.load_weights(model_or_file)
        else:
            model = clone_model(model_or_file)
            model.set_weights(model_or_file.get_weights())
        
        sparcity = 1 - pm
        sprasity_sched = tfmot.sparsity.keras.ConstantSparsity(
            sparcity, 
            0, # Do sparcity calculation in the first step
            end_step=0, # Do it only once
            frequency=10000000
        )
        
        if layers_to_prune is None:
            layers_to_prune = get_default_layers(model)
        
        prunned_model_layers = []
        for layer in model.layers:
            if layer.name in layers_to_prune:
                prunned_model_layers.append(tfmot.sparsity.keras.prune_low_magnitude(layer, sprasity_sched))
            else:
                prunned_model_layers.append(layer)
        
        pruned_model = Sequential(prunned_model_layers)
        del model
        # This is necesary to make keras calculate the mask, learning rate is 0
        initialize_pruned_model(pruned_model)
        
        return pruned_model
    
    def initialize_sparse_model(self, model_or_file, pruned_model_with_mask, pm):
        """
            Given a filename (or a model) with weights and a pruned model with its mask, returns a new model with weights in filename and pruned with mask
        """
        if type(model_or_file) == str:
            model = self.get_model()
            model.load_weights(model_or_file)
        else:
            model = clone_model(model_or_file)
            model.set_weights(model_or_file.get_weights())
        
        sparcity = 1 - pm
        sprasity_sched = tfmot.sparsity.keras.ConstantSparsity(
            sparcity, 
            0, # Do sparcity calculation in the first step
            end_step=0, 
            frequency=10000000
        )

        prunned_model_layers = []
        for i, layer in enumerate(pruned_model_with_mask.layers):
            if isinstance(layer, tfmot.sparsity.keras.pruning_wrapper.PruneLowMagnitude):
                l_weights = model.layers[i].get_weights()
                l_weights[0] = l_weights[0]*layer.pruning_vars[0][1].numpy()
                model.layers[i].set_weights(l_weights)        
                prunned_model_layers.append(tfmot.sparsity.keras.prune_low_magnitude(model.layers[i], sprasity_sched))
            else:
                prunned_model_layers.append(model.layers[i])
        prunned_model = Sequential(prunned_model_layers)
        prunned_model.compile(optimizer=optimizers.SGD(lr=0), loss='sparse_categorical_crossentropy', metrics='accuracy')
        return prunned_model
    
    def test_model_sparsity(self, model):
        for i, layer in enumerate(model.layers):
            if isinstance(layer, tfmot.sparsity.keras.pruning_wrapper.PruneLowMagnitude):
                sparcity = (layer.get_weights()[0]==0).sum()/np.product((layer.get_weights()[0]==0).shape)
                mask = layer.pruning_vars[0][1].numpy().sum()/np.product((layer.get_weights()[0]==0).shape)
                print(f'{layer.name}: {sparcity}, {mask}')
    
    def verify_mask_with_model_min_weights(self, model_, pruned_model):
        """
        Verifies that min of weights with mask 1 is higher than max of weights with mask 0
        model_ can be a filename with weights or the model
        """
        if type(model_) == str:
            model = self.get_model()
            model.load_weights(model_)
        else:
            model = model_
        for i, layer in enumerate(pruned_model.layers):
            if isinstance(layer, tfmot.sparsity.keras.pruning_wrapper.PruneLowMagnitude):
                weights_abs = np.abs(model.layers[i].get_weights()[0])
                mask = layer.pruning_vars[0][1].numpy()

                # Verify that min of weights with mask 1 is higher than max of weights with mask 0
                print(f'{layer.name}: {np.min(weights_abs[mask==1]) > np.max(weights_abs[mask==0])}, shape: {mask.shape}, sparcity: {1 - mask.sum()/np.product(mask.shape)}')
            
