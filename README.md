# CGEdit/

Code, training data, and models to accompany the paper: Masis, Neal, Green, and O'Connor. ["Corpus-Guided Contrast Sets for Morphosyntactic Feature Detection in Low-Resource English Varieties."](https://aclanthology.org/2022.fieldmatters-1.2/) Field Matters Workshop at COLING, 2022.

Contact: Tessa Masis (tmasis@cs.umass.edu), Brendan O'Connor (brenocon@cs.umass.edu)
  
  
- `data/`
  - `documentation.md`: dataset documentation 
  - `CGEdit/`
    - `AAE.tsv`, `IndE.tsv`: training sets for AAE and IndE generated via CGEdit method
  - `CGEdit-ManualGen/`
    - `AAE.tsv`, `IndE.tsv`: training sets for AAE and IndE generated via both ManualGen and CGEdit
  
- `code/`
  - `train.py`: code to fine-tune BERT-variant model
  - `eval.py`: code to evaluate fine-tuned model
  - `preprocessCORAAL.py`: code used to preprocess CORAAL transcript files for extrinsic evaluation in the paper (see Section 6); note that only interviewee speech files were used for our evaluation, not interviewer speech files
  - Note that the above scripts may require modifications in order to run on your computer
  - `tutorial.ipynb`: copy of the tutorial walking through how to use our fine-tuned models (see below, section "Using our models")
  
  
## Training models

Run the train script with the contrast set generation method ('CGEdit' or 'CGEdit-ManualGen') as the first argument and the language ('AAE' or 'IndE') as the second argument. For example: 
  
    python train.py CGEdit-ManualGen AAE 


## Evaluation

The eval script will print a prediction in [0, 1] for each linguistic feature, for each test example.

Run the eval script with the contrast set generation method used for training ('CGEdit' or 'CGEdit-ManualGen') as the first argument, the language ('AAE' or 'IndE') as the second argument, and the test set filename as the third argument (not included in this repo). For example:

    python eval.py CGEdit-ManualGen AAE testFileName


## Using our models

To access our fine-tuned model trained on the data in `CGEdit-ManualGen/AAE.tsv` for 17 African American English features, please see the Google Colab notebook [here](https://colab.research.google.com/drive/15WVU8dH90Caj5W5RaxDg_vabMfTNzjxV?usp=sharing) (or see `code/tutorial.ipynb` in this repo). This tutorial will walk you through how to access and use the model for linguistic feature detection. 

Please contact us if you would like to access our model fine-tuned on the data in `CGEdit-ManualGen/IndE.tsv` for 10 Indian English features. 
