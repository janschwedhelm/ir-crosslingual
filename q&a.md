# 1. Inducing multilingual word embedding
Questions
1. Are we allowed to use pre-trained monolingual word embeddings from FastText or do we have to train them ourselves?
2. The term supervised/unsupervised is related to the way of inducing the multilingual embedding space, isn't it?
    Supervised = we feed an external expert dictionary (like Europarl) into our model to induce the multilingual embedding
    Unsupervised = we don't feed an external expert dictionary into our model, but learn in instrinsicly in training process

Framework
1. Load fastText monolingual embeddings
2. Go from word to sentence level by using bag-of-words averaging based on the fastText monolingual embeddings
3. Align resulting matrices according to some train dictionary from Europarl (i.e. first 300k sentence pairs)
4. Train projection matrix W using algined subspaces of sentences representations and procrustes method

# 2. Supervised classification task
Framework
1. Train projection matrix according to 1 - Framework
2. Create sentence vector representations of test sentence pairs (i.e. 200k sentence pairs)
3. Map source sentence embeddings to shared vector space using trained projection matrix
4. Compute similarities between mapped source vector and target vectors (i.e. using cosine similarity) and create a ranking of highest similarities (= best translations according to our model)
5. Compute precision@k to evaluate (For which fraction, the true translation (given by Europarl) is part of the top k translations that our model determined?)

# 3. Unsupervised classification task
Questions
1. Same task as supervised task (compute similarities between mapped vectors -> get a ranking of best translations (with respect to model)) except for not using an external expert dictionary for training?
           



