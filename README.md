# Cross-Lingual Information Retrieval - Group 8
Welcome to our Information Retrieval and Web Search course project!

We have dealt with **Topic 3: Cross-Lingual Information Retrieval** (CLIR).

The underlying data needed to reproduce our reported results can be found via the following link: https://drive.google.com/file/d/1axCmZfuoNavJDXwe6jR1FoEa1j-GOZ69/view?usp=sharing

Please note that, due to memory issues, a slimmed-down version of our data is provided in the link above, only containing English-German and French-English test collections. If you want us to supply the remaining data, e.g. for evaluating language pairs containing Finnish, please contact us at any time.

Unzip it to the very same directiory you find this Readme, resulting in the following folder structure:
* ir-crosslingual
  * data
  * ir_crosslingual

You can find and reproduce the results we obtained in our report by using the Jupyter notebooks we provide under "ir_crosslingual/notebooks". They are briefly described in the following:
* **create_documents_collection.ipynb**: creates train data and test collection using German-English WikiCLIR data
* **create_global_datasets.ipynb**: creates train data and test collection using English-German Europarl data
* **create_global_test-collection.ipynb**: creates test collections of several language combinations using Europarl data
* **evaluate_documents.ipynb**: evaluates models on WikiCLIR data (see Table 5 in our report)
* **evaluate_languages.ipynb**: evaluates supervised models on Europarl data using different several different language combinations (see Table 4)
* **induce_multilingual_embeddings.ipynb**: demonstrates our approach of inducing a cross-lingual word embeddings space (please download respective files from https://fasttext.cc/docs/en/pretrained-vectors.html and put them under "data/fastText_mon_emb" in order to reproduce the results)
* **logistic_regression_{}.ipynb**: evaluates logistic regression models (see Table 2)
* **multilayer_perceptron_{}.ipynb**: evaluates MLP models (see Table 3)
* **recursive_feature_elimination.ipynb**: applies recursive feature elemination on logistic regression model
* **unsupervised_classification_languages.ipynb**: evaluates unsupervised models on Europarl data using different several different language combinations (see Table 4)
* **unsupervised_classification.ipynb**: evaluates unsupervised model on English-German Europarl data using different approaches (see Table 4)

In order to run our _web application_, simply run **app.py** which you can find at "ir_crosslingual/main".

If you have any questions please do not hesitate to contact us!


Best regards,

Rebecca, Li-Chen and Jan
