import tensorflow_model_optimization as tfmot
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers

class LTH:
    def __init__(self, get_model, compile_model):
        self.get_model = get_model
        self.compile_model = compile_model
    
    def test(self, a):
        print(a)
        
    def get_prunned_model(self, filename, layers_to_pune, X_train, y_train, pm = 0.20):
        """
        Given a filename with weights, and a list with layers to prune, returns a pruned model with correct mask
        X_train, y_train are necesary to get the mask calculated by keras (Needs a fit) pm = 1 - sparcity as mentioned in paper
        """
        sparcity = 1 - pm
        sprasity_sched = tfmot.sparsity.keras.ConstantSparsity(
            sparcity, 
            0, # Do sparcity calculation in the first step
            end_step=0, # Do it only once
            frequency=10000000
        )
        model = self.get_model()
        model.load_weights(filename)
        prunned_model_layers = []
        for layer in model.layers:
            if layer.name in layers_to_pune:
                prunned_model_layers.append(tfmot.sparsity.keras.prune_low_magnitude(layer, sprasity_sched))
            else:
                prunned_model_layers.append(layer)
        pruned_model = Sequential(prunned_model_layers)

        # This is necesary to make keras calculate the mask, learning rate is 0
        pruned_model.compile(optimizer=optimizers.SGD(lr=0), loss='sparse_categorical_crossentropy', metrics='accuracy')
        pruned_model.fit(X_train[0:1], y_train[0:1], epochs=1, verbose=0, callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
        return pruned_model
    
    def initialize_sparse_model(self, filename, pruned_model_with_mask, pm = 0.20):
        """
            given a filename with weights and a pruned model, returns a new model pruned equal but with weights in filename
        """
        sparcity = 1 - pm
        sprasity_sched = tfmot.sparsity.keras.ConstantSparsity(
            sparcity, 
            0, # Do sparcity calculation in the first step
            end_step=0, 
            frequency=10000000
        )
        model = self.get_model()
        model.load_weights(filename)
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
    
    
    def verify_mask_with_model_min_weights(self, model_, pruned_model):
        """
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
            
