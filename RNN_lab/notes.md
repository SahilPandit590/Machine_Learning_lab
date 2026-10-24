first things first what is and rnn ?
it is a neural network which happen to have a hidden state as input feature and that hidden state comes from the past output 
so txt book def is 
a rnn is a neural net which is designed to process sequential data where sequence matters 
the basic math behind the state is 
We learned this equation:
\[
Y_t=\phi(X_tW_x+Y_{t-1}W_y+b)
\]

Where:
- \(X_t\): input at the current time step.
- \(Y_{t-1}\): state from the previous time step.
- \(W_x\): weights applied to the current input.
- \(W_y\): weights applied to the previous state.
- \(b\): bias.
- \(\phi\): activation function, such as \(\tanh\).
- \(Y_t\): new state produced at the current step.


memory cells and hidden states of rnn 
the rnn has so called hidden states which can maintain previous outputs so it can be said as retaining memory 
so this is why rnn has memory cells 
but as we go further with time the gradients tend to vanish and they nearly become zero so there are further more neural nets like lstm and gru they solve the problems of memory which is caused by the rnn's 
the memory equation is given as follows :
We learned the general idea:
\[
h_t=f(h_{t-1},x_t)
\]

- \(h_t\): current hidden state, representing the information retained up to the current step.
- \(h_{t-1}\): previous hidden state.
- \(x_t\): current input.
- \(f\): the function that updates the state.
the ways in which rnn can process the data is given below :
Four ways RNNs can process inputs and outputs
We learned that RNN architectures can be organized according to the shapes of their inputs and outputs.









1. Sequence-to-sequence

A sequence goes in, and a sequence comes out.

Example: predicting a series of future temperatures from previous temperatures.

2. Sequence-to-vector

A sequence produces one output.

Example: reading a movie review and predicting whether its sentiment is positive or negative.
3. Vector-to-sequence

One input vector produces a sequence.

Example: using an image representation to generate a caption.

4. Encoder-decoder

An encoder processes the input sequence into a representation, and a decoder uses it to generate an output sequence.

Example: translating an English sentence into Japanese.

so the question arises how do we train a rnn ?
there are 4 steps 
the forward pass in which the rnn 
generates the output for the inputs 
the cost calculation :
here the cost or the loss is calculated 
the backpropagation through time:
the rnn store hidden states for each of the neuron in a separate array so by updating those and recalculating the next output for the next neurons and hidden states for them the rnn handles the back propagation through time 
update the parameters : 
After calculating gradients, the optimizer updates the weights and biases to reduce the loss.

For a simple gradient-descent update:
\[
W_{\text{new}}=W_{\text{old}}-\eta\frac{\partial C}{\partial W}
\]

Trend

A long-term increase or decrease.
Seasonality

A pattern that repeats at regular intervals