# Food Category Predictor - FDA Web Tool

Demo: http://people.csail.mit.edu/yygu/web_tool

This tool helps generate food category predictions based on the Concept Unique Identifier (CUI) of an ingredient or adulterant. We extract the hierarchy based on the CUI, and run this hierarchy through a neural network to generate the predictions. The data for the hierarchies is extracted from the [UMLS Metathesaurus](https://www.nlm.nih.gov/pubs/factsheets/umlsmeta.html). 

## Requirements

The tool requires Python 2.7. In addition, you will need to install the following packages:

- Flask (0.11.1)
- pandas (0.17.1)
- numpy (1.10.4)
- Theano (0.8.0)

The number in parenthesis states the package version that was used during testing of this tool. A newer or older version might work, but is not guaranteed.

## Installation

This was written for (and tested on) Mac OS X and Linux. Some modification to the installation instructions might be necessary to get it to run on Windows.

1. Install all the required packages above: `pip install -r requirements.txt`.
2. Download two files containing the Metathesaurus data: `wget https://www.dropbox.com/s/7yntvewhb8uxrlc/mrhier.h5 https://www.dropbox.com/s/9vishxtcdpft5hg/mrconso.h5`. Alternatively, you can download [File 1](https://www.dropbox.com/s/7yntvewhb8uxrlc/mrhier.h5?dl=1) (1GB) and [File 2](https://www.dropbox.com/s/9vishxtcdpft5hg/mrconso.h5?dl=1) (500MB) manually.
2. Go to this directory in a terminal. Type `export FLASK_APP=main.py`
3. Type `flask run` to run locally and `flask run --host=0.0.0.0` to run an externally visible server. See the Flask documentation for more options on how to deploy the server: http://flask.pocoo.org/docs/0.11/quickstart/
4. Visit http://0.0.0.0:5000. It's ready for use! (If you use an externally visible server, you can also replace "0.0.0.0" with the IP address of the machine to view it from any machine. However, this is subject to router/firewall restrictions.).

## How to use

### Concept Unique Identifier (CUI)

The Concept Unique Identifier (CUI) is a value assigned to a "concept" in the [UMLS Metathesaurus](https://www.nlm.nih.gov/research/umls/new_users/online_learning/Meta_005.html). It is used to refer any ingredient, adulterant, chemical, or substance that exists in the Metathesaurus. We will use this as the identifier from which the model will be able to make predictions.

#### Choosing a CUI

Using the tool is very easy. Say you have an ingredient/adulterant that you would like to find the CUI for. You would need to visit the [NCI Metathesaurus](https://ncimeta.nci.nih.gov/ncimbrowser/) web page, and enter the ingredient/adulterant in the search bar. Select the result with the best match (Look carefully that the "Semantic Type" makes sense for your substance). The CUI should be immediately available once you click on the result.

### Hierarchy

This tool relies on the hierarchy of the CUI to make accurate predictions. Therefore, a hierarchy must exist for a prediction to exist. The hierarchy can be found as "Parent Concepts" under the "Relationships" tab of a CUI page. You can click on each parent concept to take you to the CUI of the parent, allowing you to essentially traverse the hierarchy tree.

We use the hierarchy relationships from four data sources: Systematized Nomenclature of Medicine - Clinical Terms (SNOMEDCT), National Cancer Institute (NCI), National Drug File - Reference Terminology (NDFRT), and Management Sciences for Health (MSH).

Sometimes, a CUI has a hierarchy on the NCI Metathesaurus page, but the web tool will say that no hierarchy exists in the database. This is because the concepts within the hierarchy were not seen during the training of the model. Hence, the model is not able to make a prediction from this. 

If for whatever reason a hierarchy is unable to be found, we recommend that you choose another CUI for the substance in question. If there are multiple CUI candidates for the same substance, there might be one that would generate better results. If not, then we are out of luck - the model is unable to make a prediction for this substance.

#### Nodes

Each hierarchy consists of multiple nodes (which themselves are a CUI). For example, the following hierarchy for "oats" has 5 nodes: oats → grain → foods → dietary substance → substance. The model uses these nodes to make predictions. 

For each prediction, the tool displays the list of all nodes used by the model. If the list of nodes seems too general, then the model is unable to find a specific enough representation for prediction. In order to make a useful prediction from a node (e.g. "grain"), the model must have seen this node during training. Even if a hierarchy exists in the Metathesaurus, the model is sometimes unable to make a good prediction because it has not seen many of the nodes in the hierarchy. A warning will show up when the number of nodes used in the prediction is low.

### Predictions

The model outputs a table of the top 20 most likely food product categories (out of 131). The score represents the following: Given all food products which contains this substance, what fraction of them belong in this category. Therefore, the sum of the scores of all 131 categories is 1. For a more general purpose analysis, the score can be ignored in favor of the rank.

_Note: When there is not enough data for the model to make an accurate prediction, the model will often favor certain categories (e.g. snack, energy & granola bars and frozen appetizers & hors d'oeuvres) because there were a lot of products in this category during training._

## Model

We trained a [multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) to predict likely food product categories from the hierarchy of a substance. The model was trained on the 5000 common ingredients from [FoodEssentials](http://developer.foodessentials.com/) as well as 439 adulterants from [Rapid Alert System for Food and Feed](ec.europa.eu/food/safety/rasff/index_en.htm). All the ingredients from FoodEssentials come from food products with a valid UPC code, so certain product categories that do not contain UPC codes (such as fresh fruit, produce, and meats) will not show up in the predictions. We used a hidden dimension of 50 and a softmax layer at the output. This is a classification problem, where a substance must be classified to belong in one of 131 food product categories. A list of all 131 categories can be seen in `categories.txt`.

For any questions or issues, contact Youyang Gu (yygu@csail.mit.edu).
