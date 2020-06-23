import numpy as np
from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras import optimizers
import tensorflow as tf
from matplotlib import pyplot as plt
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
    

from matplotlib.colors import ListedColormap
from tensorflow.keras.utils import to_categorical

def get_custom_cmap(Ri=0, Gi=0, Bi=0, alpha=0.8, almost_white=0.99):
    N_c = 256
    N_c_col = int(N_c*almost_white)
    marg = 0
    R = np.hstack([np.zeros((N_c-N_c_col)-marg), np.linspace(1, Ri, N_c_col+marg)])
    G = np.hstack([np.zeros((N_c-N_c_col)-marg), np.linspace(1, Gi, N_c_col+marg)])
    B = np.hstack([np.zeros((N_c-N_c_col)-marg), np.linspace(1, Bi, N_c_col+marg)])
    A = np.hstack([np.zeros((N_c-N_c_col)-marg), alpha*np.ones(N_c_col+marg)])
    custom_map = ListedColormap(np.vstack([R,G,B,A]).T)
    return custom_map

def plot_MC_boundaries_keras(X_train, y_train, score, probability_func, degree=None, bias=False, 
                             mesh_res = 300, ax = None, margin=0.5, color_index = 0, normalize = False, alpha=0.5, almost_white=0.99,
                             my_colors=[[0,0,0.5], [0,0.5,0], [0.5,0,0], [0.5,0.5,0.5], [0,0.5,0.5]]):
    y_train_cat = to_categorical(y_train)
#     if (y_train_cat_aux.shape[1] > 2):
#         y_train_cat = y_train_cat_aux
#     else:
#         y_train_cat = y_train
    X = X_train
    margin_x = (X[:, 0].max() - X[:, 0].min())*0.05
    margin_y = (X[:, 1].max() - X[:, 1].min())*0.05
    x_min, x_max = X[:, 0].min() - margin_x, X[:, 0].max() + margin_x
    y_min, y_max = X[:, 1].min() - margin_y, X[:, 1].max() + margin_y
    hx = (x_max-x_min)/mesh_res
    hy = (y_max-y_min)/mesh_res
    x_domain = np.arange(x_min, x_max, hx)
    y_domain = np.arange(y_min, y_max, hy)
    xx, yy = np.meshgrid(x_domain, y_domain)
    
    
    if ax is None:
        ax = plt.subplot(1, 1, 1)
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    
    if degree is not None:
        polynomial_set = get_polynimial_set(np.c_[xx.ravel(), yy.ravel()], degree = degree, bias=bias)
        Zaux = probability_func(polynomial_set)
    else:
        Zaux = probability_func(np.c_[xx.ravel(), yy.ravel()])
        # Z = Z_aux[:, 1]
    
    # if Zaux.shape[1] == 2:
        # Es un polinomio
        # Z = Zaux[:, 1]
    # else:
        # No es un polinomio
        # Z = Zaux[:, 2]

    # Put the result into a color plot
    
    if normalize:
        Zaux = (Zaux.T/Zaux.sum(axis=1)).T
    
    cm_borders = ListedColormap(["#FFFFFFFF", "#000000"])
    # my_colors = cm.cividis.colors # 
    # my_colors = [list(x) for x in list(mpl.colors.BASE_COLORS.values())]
    cat_order = len(y_train_cat.shape)
    if cat_order>1:
        Z_reshaped = Zaux.reshape(xx.shape[0], xx.shape[1], y_train_cat.shape[1])
        for i in range(Z_reshaped.shape[2]):
            my_cmap = get_custom_cmap(my_colors[i][0],my_colors[i][1],my_colors[i][2], alpha=alpha, almost_white=almost_white)
            Z = Z_reshaped[:,:,i]    

#             cf = ax.contourf(xx, yy,
#                              Z,
#                              50, 
#                              vmin = 0,
#                              vmax = 1,
#                              cmap=my_cmap, 
#                             )

            Z_class = np.argmax(Z_reshaped, axis=2)
            Z_selected = (Z_class == i)*Z

            cf = ax.contourf(xx, yy, Z_selected, 50, 
                                 vmin = 0,
                                 vmax = 1,
                                 cmap=my_cmap, 
                                )
            ax.scatter(X_train[:, 0], X_train[:, 1], 
               c=y_train, 
               cmap=ListedColormap(my_colors),
               edgecolors='k', 
               s=100)
            thres = 1/Z_reshaped.shape[2]
            ax.contour(xx, yy, Z_selected, (thres,), colors='k', linewidths=0.5)
    else:
        Z_reshaped = Zaux.reshape(xx.shape[0], xx.shape[1])
        my_cmap = get_custom_cmap(my_colors[color_index][0],my_colors[color_index][1],my_colors[color_index][2], alpha=0.5)
        cf = ax.contourf(xx, yy,
                             Z_reshaped,
                             256, 
                             vmin = 0,
                             vmax = 1,
                             cmap=my_cmap, 
                            )
        ax.scatter(X_train[:, 0], X_train[:, 1], 
               c=y_train, 
               # cmap=ListedColormap(my_colors[color_index]),
               edgecolors='k', 
               s=100)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    #ax.set_xticks(())
    #ax.set_yticks(())
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
            size=40, horizontalalignment='right')
    return xx, yy, Z_reshaped    
    
# class LTH:
#     def __init__(self, get_model):
#         self.get_model = get_model
    
#     def test(self, a):
#         print(a)
    
    
    
    
        
    
#     def get_prunned_model(self, model_or_file, pm=0.5, X_train=None, y_train=None, layers_to_prune=None):
#         """
#         Given a filename with weights, and a list with layers to prune, returns a pruned model with correct mask
#         X_train, y_train are necesary to get the mask calculated by keras (Needs a fit) pm = 1 - sparcity as mentioned in paper
#         """
        
#         if type(model_or_file) == str:
#             model = self.get_model()
#             model.load_weights(model_or_file)
#         else:
#             model = clone_model(model_or_file)
#             model.set_weights(model_or_file.get_weights())
        
#         sparcity = 1 - pm
#         sprasity_sched = ConstantSparsity(
#             sparcity, 
#             0, # Do sparcity calculation in the first step
#             end_step=0, # Do it only once
#             frequency=10000000
#         )
        
#         if layers_to_prune is None:
#             layers_to_prune = get_default_layers(model)
        
#         prunned_model_layers = []
#         for layer in model.layers:
#             if layer.name in layers_to_prune:
#                 prunned_model_layers.append(prune_low_magnitude(layer, sprasity_sched))
#             else:
#                 prunned_model_layers.append(layer)
        
#         pruned_model = Sequential(prunned_model_layers)
#         del model
#         # This is necesary to make keras calculate the mask, learning rate is 0
#         initialize_pruned_model(pruned_model)
        
#         return pruned_model
    
#     def initialize_sparse_model(self, model_or_file, pruned_model_with_mask, pm):
#         """
#             Given a filename (or a model) with weights and a pruned model with its mask, returns a new model with weights in filename and pruned with mask
#         """
#         if type(model_or_file) == str:
#             model = self.get_model()
#             model.load_weights(model_or_file)
#         else:
#             model = clone_model(model_or_file)
#             model.set_weights(model_or_file.get_weights())
        
#         sparcity = 1 - pm
#         sprasity_sched = ConstantSparsity(
#             sparcity, 
#             0, # Do sparcity calculation in the first step
#             end_step=0, 
#             frequency=10000000
#         )

#         prunned_model_layers = []
#         for i, layer in enumerate(pruned_model_with_mask.layers):
#             if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
#                 l_weights = model.layers[i].get_weights()
#                 l_weights[0] = l_weights[0]*layer.pruning_vars[0][1].numpy()
#                 model.layers[i].set_weights(l_weights)        
#                 prunned_model_layers.append(prune_low_magnitude(model.layers[i], sprasity_sched))
#             else:
#                 prunned_model_layers.append(model.layers[i])
#         prunned_model = Sequential(prunned_model_layers)
#         prunned_model.compile(optimizer=optimizers.SGD(lr=0), loss='sparse_categorical_crossentropy', metrics='accuracy')
#         return prunned_model
    
#     def test_model_sparsity(self, model):
#         for i, layer in enumerate(model.layers):
#             if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
#                 sparcity = (layer.get_weights()[0]==0).sum()/np.product((layer.get_weights()[0]==0).shape)
#                 mask = layer.pruning_vars[0][1].numpy().sum()/np.product((layer.get_weights()[0]==0).shape)
#                 print(f'{layer.name}: {sparcity}, {mask}')
    
#     def verify_mask_with_model_min_weights(self, model_, pruned_model):
#         """
#         Verifies that min of weights with mask 1 is higher than max of weights with mask 0
#         model_ can be a filename with weights or the model
#         """
#         if type(model_) == str:
#             model = self.get_model()
#             model.load_weights(model_)
#         else:
#             model = model_
#         for i, layer in enumerate(pruned_model.layers):
#             if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
#                 weights_abs = np.abs(model.layers[i].get_weights()[0])
#                 mask = layer.pruning_vars[0][1].numpy()

#                 # Verify that min of weights with mask 1 is higher than max of weights with mask 0
#                 print(f'{layer.name}: {np.min(weights_abs[mask==1]) > np.max(weights_abs[mask==0])}, shape: {mask.shape}, sparcity: {1 - mask.sum()/np.product(mask.shape)}')
            
