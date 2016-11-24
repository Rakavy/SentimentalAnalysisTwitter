# SentimentalAnalysisTwitter

We are building a natural language processor which will determine how polarizing topics are by parsing through twitter hashtags and seeing the trends in positivity and negativity in those terms.

Notes:

Ok, so I spent 3 hours getting a good idea of how the tutorial we are basing our stuff on is working. I was really hoping to make more progress, but alas I wasn't very productive

Unfortunately it appears that the tutorial is in fact using a word embedding layer. The good news is that it doesn't appear to be very complicated.

-The training data must be split between training set and validation set (5% of data for validation is their default). 

-Instead of doing every training step using the entire training set, they use random subsets called mini-batches, which are determined using the get_Mini_Batches function, which returns a list of tuples (batch index, list of training sample indices).

-I rewrote/copy-pasted the tutorial's functions to initialize weights matrices and biases, both in the embedding, LSTM and classifier (the last one, which decides the output) layers. The LSTM uses orthogonal initialization, which is based on Singular Value Decomposition. No clue why it is necessary.

-The init functions will be called at the beginning of the main training function

-We also have to write functions to load/read the weights and biases after training, to actually use the network

-To end on a good note, I'm pretty sure I got a good grasp of how to proceed and how everything should work together. 
