import numpy as np
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
import tensorflow as tf
from tensorflow_model_optimization.sparsity.keras import prune_low_magnitude, ConstantSparsity, UpdatePruningStep, strip_pruning, pruning_wrapper


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
        
    pruned_model.fit(X_train[0:1], y_train[0:1], epochs=1, verbose=0, callbacks=[UpdatePruningStep()])
    return pruned_model

def is_pruned(model):
    for layer in model.layers:
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            return True
    return False

def apply_wrapper_to_layer(model_to_clone, layer_names, wrapper, sprasity_sched, clone=False):
    if clone:
        model = clone_model(model_to_clone)
        model.set_weights(model_to_clone.get_weights())
    else:
        model = model_to_clone
        
    layers = [l for l in model.layers]
    
    if not isinstance(model.layers[0], tf.python.keras.engine.input_layer.InputLayer):
        prunned_model_layers = []
        for layer in model.layers:
            if layer.name in layer_names:
                prunned_model_layers.append(wrapper(layer, sprasity_sched))
            else:
                prunned_model_layers.append(layer)
        new_model = Sequential(prunned_model_layers)
    else:
        in_shape = model.layers[0].input_shape[0]
        layers = layers[1:]
        inp = Input(shape=in_shape[1:])
        x = inp
        for layer in layers:
            if layer.name in layer_names:
                x = wrapper(layer, sprasity_sched)(x)
            else:
                x = layer(x)

        new_model = Model(inp, x)
    return new_model

def prune_and_initilize(trained_model, pm, initial_weights, layers_to_prune=None):
    sparcity = 1 - pm
    sprasity_sched = ConstantSparsity(
        sparcity, 
        0, # Do sparcity calculation in the first step
        end_step=0, # Do it only once
        frequency=10000000
    )
    
    model = clone_model(trained_model)
    model.set_weights(trained_model.get_weights())
    
    if is_pruned(model):
        model = strip_pruning(model)

    if layers_to_prune is None:
        layers_to_prune = get_default_layers(model)

#     prunned_model_layers = []
#     for layer in model.layers:
#         if layer.name in layers_to_prune:
#             prunned_model_layers.append(prune_low_magnitude(layer, sprasity_sched))
#         else:
#             prunned_model_layers.append(layer)

#     trained_pruned_model = Sequential(prunned_model_layers)
    
    trained_pruned_model = apply_wrapper_to_layer(model, layers_to_prune, prune_low_magnitude, sprasity_sched, clone=False)
    # Calculates mask
    initialize_pruned_model(trained_pruned_model)
        
    model.load_weights(initial_weights)
    prunned_model_layers = []
    for i, layer in enumerate(trained_pruned_model.layers):
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            l_weights = model.layers[i].get_weights()
            l_weights[0] = l_weights[0]*layer.pruning_vars[0][1].numpy()
            model.layers[i].set_weights(l_weights)        
            prunned_model_layers.append(prune_low_magnitude(model.layers[i], sprasity_sched))
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
    sprasity_sched = ConstantSparsity(
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
            prunned_model_layers.append(prune_low_magnitude(layer, sprasity_sched))
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
    sprasity_sched = ConstantSparsity(
        sparcity, 
        0, # Do sparcity calculation in the first step
        end_step=0, 
        frequency=10000000
    )

    prunned_model_layers = []
    for i, layer in enumerate(pruned_model_with_mask.layers):
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            l_weights = model.layers[i].get_weights()
            l_weights[0] = l_weights[0]*layer.pruning_vars[0][1].numpy()
            model.layers[i].set_weights(l_weights)        
            prunned_model_layers.append(prune_low_magnitude(model.layers[i], sprasity_sched))
        else:
            prunned_model_layers.append(model.layers[i])
    prunned_model = Sequential(prunned_model_layers)
    prunned_model.compile(optimizer=optimizers.SGD(lr=0), loss='sparse_categorical_crossentropy', metrics='accuracy')
    return prunned_model
    
def test_model_sparsity(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
            sparcity = (layer.get_weights()[0]==0).sum()/np.product((layer.get_weights()[0]==0).shape)
            mask = layer.pruning_vars[0][1].numpy().sum()/np.product((layer.get_weights()[0]==0).shape)
            print(f'{layer.name}: {sparcity}, {mask}')
        elif isinstance(layer, tf.keras.layers.Dense) or isinstance(layer, tf.keras.layers.Conv2D):
            sparcity = (layer.get_weights()[0]==0).sum()/np.product((layer.get_weights()[0]==0).shape)
            print(f'{layer.name}: {sparcity} - no mask')
    
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
        sprasity_sched = ConstantSparsity(
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
                prunned_model_layers.append(prune_low_magnitude(layer, sprasity_sched))
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
        sprasity_sched = ConstantSparsity(
            sparcity, 
            0, # Do sparcity calculation in the first step
            end_step=0, 
            frequency=10000000
        )

        prunned_model_layers = []
        for i, layer in enumerate(pruned_model_with_mask.layers):
            if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
                l_weights = model.layers[i].get_weights()
                l_weights[0] = l_weights[0]*layer.pruning_vars[0][1].numpy()
                model.layers[i].set_weights(l_weights)        
                prunned_model_layers.append(prune_low_magnitude(model.layers[i], sprasity_sched))
            else:
                prunned_model_layers.append(model.layers[i])
        prunned_model = Sequential(prunned_model_layers)
        prunned_model.compile(optimizer=optimizers.SGD(lr=0), loss='sparse_categorical_crossentropy', metrics='accuracy')
        return prunned_model
    
    def test_model_sparsity(self, model):
        for i, layer in enumerate(model.layers):
            if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
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
            if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
                weights_abs = np.abs(model.layers[i].get_weights()[0])
                mask = layer.pruning_vars[0][1].numpy()

                # Verify that min of weights with mask 1 is higher than max of weights with mask 0
                print(f'{layer.name}: {np.min(weights_abs[mask==1]) > np.max(weights_abs[mask==0])}, shape: {mask.shape}, sparcity: {1 - mask.sum()/np.product(mask.shape)}')
            
