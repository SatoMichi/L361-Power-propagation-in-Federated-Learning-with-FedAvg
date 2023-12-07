# Power-propagation-in-Federated-Learning-with-FedAvg
This is a repository for final project of L361 Federated Learning course in MPhil Advanced Computer Science.

## Abstruct
This project investigated the impact of power-propagation (raising the absolute value of a parameter to a power during inference) with different powers upon the sparsity of Federated Learning (FL) models. More specifically, we tried to obtain a sparse model on the Server side aggregated global model while applying power-propagation in the training process of the client’s local model. An simple Multiple Layer Perceptron (MLP) was used for this task.  

Please read *https://arxiv.org/abs/2110.00296* for understanding about power-propagation.  

## Sparsity of the model
The **sparsity** of the model is one of the essential concepts for machine learning (or more specifically deep learning). Sparsity in a machine learning model is usually referring the sparsity of the weight matrix in the model architecture. Simply speaking, a sparse model means that the number of zero weight (or in other words, the weight which is insignificant to the model’s output) dominates a large part of the model weight parameters.

Sparsity is an essential property for machine learning or deep learning model since it is known that the sparse model will have less time for computation and less size of the model. It is also known that the sparse model will prevent the phenomena called [catastrophic forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference) (which is a phenomena where the model forget the parameter once the new data is added and trained).

## What is Federated Learning?
You can look at the [Federated Learning](https://en.wikipedia.org/wiki/Federated_learning) in Wikipedia.  
The brief idea is that the model will not trained on central server. Instead the model will trained on each device (client) and the trained model will be aggregated on the centrl server. Following are simplified process;
1. Server prepare the model *M* and distribute it to each device.
2. Each device conduct training of distributed model *M*, and obtain *M'*.
3. Server collect trained *M'* from each device and aggregate these model to obtain new model *M**.
4. Server distribute obtained new model *M** to each device.
5. Repeat process 2-4.

![1_WgNu6ZtQ-3vXRtMdvLYzpw](https://github.com/SatoMichi/L361-Power-propagation-in-Federated-Learning-with-FedAvg/assets/44910734/3c995f3b-570a-4849-9e94-9816995f957c)
Image from https://ankitasinha0811.medium.com/beginners-guide-to-federated-learning-d529557a1b1e

Different from traditional model training process which require the all data to be collected to central server, FL will allow each device to keep thier data in thier storage and do not need to be sent to server. This will have positive effect on protecting privacy information. 

## Code
Some code and implementation are from 
- https://github.com/mysistinechapel/powerprop
- Class lab notebooks of the L361 course

We will briefly explain what each code file are used for;
- dataset.py: used to split FEMNIST images data and generate dataset for training, validation and testing. FEMINIST data need to be download for running experiments.
- model.py: defined the simple MLP model used for our task.
- client.py: define the clients of the FL. Usually these are individual devices which have its own data inside.
- server.py: define the server of the FL. Typically, the server will aggreagte each models trained on different clients.

You will need to create result file and model file for saving the experiments results. If you want to run the experiment, we will suggest using small epochs, clients_per_round, and rounds since the experiment will took time

## Experimets
### EXPERIMENT01: MODEL IN CLIENT
Code: *single_client_model.ipynb*  
We first applied power-propagation at the client level to observe the sparsity obtained. We used the full data set of FEMNIST for this experiments. Constructed Client and passed powerpropagation MLP model, then conducted the training and observed the sparsity of the model obtained.

### EXPERIMENT02: POWER-PROPAGATION WITH FL
Code: *FL_experimet01.ipynb*, *FL_experimet02.ipynb*, *FL_experimet03.ipynb*  
Next, we conducted the experiments with the FL setting. Following steps are the brief process of this experiment;
1. Server samples clients for this round of federated learning.
2. The server distributes the global model to these clients.
3. Each sampled client will conduct the training of the distributed model on their local data.
4. Once the training of the clients is finished, the server will collect the updated weight of the local model in each client.
5. The server will aggregate these collected weight parameters based on FedAvg methods and update the global model. At this time, the global model will be evaluated on the whole data of the FEMNIST for recording the performance over the rounds.
6. Repeat steps 1-5 for a fixed number of rounds.

## Conclusion
Through this research, we are able to prove our assumption that the larger value of power *α* will lead to a more sparse model due to the power-propagation method (this is mainly proved in Experiment 01).

The number of rounds will be more important than the number of clients per round. In fact, our experiment result suggests fewer clients per round will lead to better sparsity of the final global model due to the nature of FedAvg.