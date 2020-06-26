# MJLS_Learning

This repository contains sample code to run the experiments in the following papers:

- Convergence Guarantees of Policy Optimization Methods for Markovian Jump Linear Systems (accepted to ACC 2020, arXiv:2002.04090);
- Policy Learning of MDPs with Mixed Continuous/Discrete Variables: A Case Study on  Model-Free Control of Markovian Jump Systems (accepted to L4DC 2020, arXiv:2006.03116),

both authored by Joao Paulo Jansch-Porto, Bin Hu, and Geir Dullerud.

### Usage ###

The main_small.cpp is optimized (for speed) for the small scale example in the second paper, while main_large.cpp can be used for all other systems.


### Requirements ###

We have the following code dependencies:

-Eigen 3.3 (or greater);

-MATLAB mat and mx libraries (we use MATLAB to generate the system matrices and to return the expected costs);

-C++11 (or greater).

The code was tested both on Windows (using MSVC17) and Linux (using g++ version 7.4).
