# TensorFlow With The Latest Papers Implemented

###Implementation of RNN and NLP Related Neural Network Papers

Currently Implemented Papers:

* Kronecker Highway RNN
* Delta RNN
* Highway Networks
* Recurrent Highway Networks
* Multiplicative Integration Within RNNs
* Recurrent Dropout
* Layer Normalization
* Layer Normalization & Multiplicative Integration
* LSTM With Multiple Memory Arrays
* Minimal Gated Unit RNN
* Residual Connections Within Stacked RNNs
* GRU Mutants
* Weight Tying

More Papers to come as they are published. If you have any requests, please use the issues section. 

###Contact Information:

skype: lea vesbr eat he thisisjunk(eliminate all spaces and ignore junk part)

email: sh a hn s [at ] m ail.u c.ed u thisisjunk(eliminate all spaces and ignore junk part)

linkedin: www.linkedin.com/in/NickShahML


### To Test These New Papers

If you would like to test these new features, you can:

`python ptb_word_lm.py`

Simply modify the `rnn_cell` variable under the `PTBModel`

Please run with TensorFlow 0.10 or higher

### Kronecker RNN

https://arxiv.org/pdf/1705.10142.pdf

Uses Kronecker Products to reparameterize weight matrices. Implemented 2x2 kronecker product.

Warning: there is no l2 soft unitary constraint implemented. This is on my to do. 

```python
import rnn_cell_modern

rnn_cell = rnn_cell_modern.HighwayRNN(inputs, num_units = 1024, use_kronecker_reparameterization=True)

# To Call
output, new_state = rnn_cell(inputs, state)
```


### Delta RNN

https://arxiv.org/pdf/1703.08864.pdf

Light yet powerful RNN that beats LSTM on multiple tests. Implemented second order delta rnn cell

```python
import rnn_cell_modern

rnn_cell = rnn_cell_modern.Delta_RNN(inputs, num_units = 1024)

# To Call
output, new_state = rnn_cell(inputs, state)
```

### Highway Networks

https://arxiv.org/abs/1505.00387

Allows greater depth of neural network freely without penalty from upper layers. Ensures shortcut connections within deeper layers.

Note, there is an optional flag `use_inputs_on_each_layer` that is boolean. Turning this option to False saves network parameters but also may not yield most optimal results. If you would like to replicate the paper, leave this option as False.
```python
import highway_networks_modern

output = highway_networks_modern.highway(inputs, num_layers = 3)
```



### Recurrent Highway Networks 

http://arxiv.org/abs/1607.03474

Allows multiple stacking of layers within one cell to increase depth per timestep. 

```python
import rnn_cell_modern

cell = rnn_cell_modern.HighwayRNNCell(num_units, num_highway_layers = 3)
```


### Multiplicative Integration Within RNNs

https://arxiv.org/abs/1606.06630

Allows faster convergence within RNNs by utilizing the combination of two separate weight matrices in a multiplicative setting

```python

import rnn_cell_mulint_modern

cell = rnn_cell_mulint_modern.BasicRNNCell_MulInt(num_units)
#OR
cell = rnn_cell_mulint_modern.GRUCell_MulInt(num_units)
#OR
cell = rnn_cell_mulint_modern.BasicLSTMCell_MulInt(num_units)
#OR
cell = rnn_cell_mulint_modern.HighwayRNNCell_MulInt(num_units, num_highway_layers = 3)
```


### Recurrent Dropout Without Memory Loss

http://arxiv.org/pdf/1603.05118v1.pdf

Implement recurrent dropout within multiplicative integration rnn cells. Will allow rnn cell's memory to be more versatile. 

```python
import rnn_cell_mulint_modern

#be sure to change recurrent_dropout_value to 1.0 during testing or validation
#alternatively, you can set the is_training argument to False during testing or validation but this requires the reconstruction of the model

cell = rnn_cell_mulint_modern.BasicLSTMCell_MulInt(num_units, use_recurrent_dropout = True, recurrent_dropout_value = 0.90)
#OR
cell = rnn_cell_mulint_modern.GRUCell_MulInt(num_units,  use_recurrent_dropout = True, recurrent_dropout_value = 0.90)
#OR
cell = rnn_cell_mulint_modern.BasicLSTMCell_MulInt(num_units, use_recurrent_dropout = True, recurrent_dropout_value = 0.90)
#OR
cell = rnn_cell_mulint_modern.HighwayRNNCell_MulInt(num_units, num_highway_layers = 3,  use_recurrent_dropout = True, recurrent_dropout_value = 0.90)

```


### Layer Normalization
http://arxiv.org/abs/1607.06450

Layer normalization promises faster convergence and lower perplexities. With layer normalization you do not need to change any settings if you're training or testing.

Note: It seems that the GRU implementation does not converge currently. I've found that it does converge if you only apply LN terms to the first two r and u matrices. 

```python
import rnn_cell_layernorm_modern

rnn_cell = rnn_cell_layernorm_modern.BasicLSTMCell_LayerNorm(size)
#OR
rnn_cell = rnn_cell_layernorm_modern.GRUCell_LayerNorm(size)
#OR
rnn_cell = rnn_cell_layernorm_modern.HighwayRNNCell_LayerNorm(size)
```

### Layer Normalization & Multiplicative Integration
http://arxiv.org/abs/1607.06450

Layer normalization is currently implemented within a multiplicative integration context. If there are requests for a vanilla implementation for layer normalization please let me know. With layer normalization you do not need to change any settings if you're training or testing.

As a warning, this implementation is experimental and may not produce favorable training results.

```python
import rnn_cell_mulint_layernorm_modern

rnn_cell = rnn_cell_mulint_layernorm_modern.BasicLSTMCell_MulInt_LayerNorm(size)
#OR
rnn_cell = rnn_cell_mulint_layernorm_modern.GRUCell_MulInt_LayerNorm(size)
#OR
rnn_cell = rnn_cell_mulint_layernorm_modern.HighwayRNNCell_MulInt_LayerNorm(size)
```

### LSTM With Multiple Memory Arrays

Implementation of Recurrent Memory Array Structures by Kamil Rocki
https://arxiv.org/abs/1607.03085

Idea is to build more complex memory structures within one single layer rather than stacking multiple layers of RNNs.

When using this type of cell, it is recommended to only use one single layer and then increase the number of memory cells. 

Within this implementation you can also choose to use or not use:
- multiplicative integration
- recurrent dropout
- layer normalization

I have found multiplicative integration to help and have not extensively tested the other options.

```python
import rnn_cell_modern

rnn_cell = rnn_cell_modern.LSTMCell_MemoryArray(size, num_memory_arrays = 2, 
	use_multiplicative_integration = True, use_recurrent_dropout = False, use_layer_normalization = False)
```

### Minimal Gated Unit Recurrent Neural Network

Implementation of Minimal Gated Unit by Zhou
http://arxiv.org/abs/1603.09420

This minimal RNN can match the performance of GRU and has 33% less parameters. As a result, it computes about 20% faster on a Titan X compared to a same sized GRU. Very optimal RNN for a quick test of dataset. This implementation also has options for:

- Multiplicative Integration
- Recurrent Dropout
- Forget Bias Initialization

The forget_bias_initialization can be experimented around with. The authors do not specify what value works best.

```python
import rnn_cell_modern

cell = rnn_cell_modern.MGUCell(num_units, use_multiplicative_integration = True, use_recurrent_dropout = False, forget_bias_initialization = 1.0)

```

### Residual Connections Within Stacked RNNs

Implementation of Residual Connections as used in Google's Translation System:
https://arxiv.org/abs/1609.08144

This will allow for stacking multiple RNN layers with improved result due to residual connections. Inputs of the previous layer and added to the current input of the current layer.

```python
import rnn_wrappers_modern

cell = DesiredRNNCell(num_units)

stacked_cells = rnn_wrappers_modern.MultiRNNCell([cell]*4, use_residual_connections = True) #stacks 4 layers
```


### GRU Mutants

http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf

Mutants of GRU that may work better in different scenarios:

```python
import rnn_cell_modern

cell = rnn_cell_modern.JZS1Cell(num_units)
#Or
cell = rnn_cell_modern.JZS2Cell(num_units)
#Or
cell = rnn_cell_modern.JZS3Cell(num_units)
```

### Weight Tying

"Using the Output Embedding to Improve Language Models" by Press & Wolf
https://arxiv.org/abs/1608.05859

Tying the input word embeding to the softmax matrix. Because of the similarities between the input embedding and the softmax matrix (AKA the output embedding), setting them to be equal improves preplexity while reducing the number of parameters in the model. 
```python
softmax_w = tf.transpose(embedding)
```
