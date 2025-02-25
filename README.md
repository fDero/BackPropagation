# BackPropagation
A very simple implementation of the back-propagation algorithm for neural networks training, 
written in python, from scratch (no-frameworks).

### What is this repo?
This codebase takes heavy inspiration from [micrograd](https://github.com/karpathy/micrograd) 
which is a python toy-project that was created to serve as an example. I rewrote the entire 
thing as a learning exercise.

### How is this implementation any better than others?
While micrograd itself stores temporary data in each neuron 
making it possible to cause a bug by not resetting every neuron before re-iterating during 
training, my approach was to use recursion to return everything to the caller without storing 
any temporary data, indeed making the API less error-prone. <br/>

The API of this codebase also makes impossible to consider the gradients of immutable 
input parameters during training, since, by design, immutable parameters have their own 
specialized implementation of the method that computes gradients, and it returns nothing.
