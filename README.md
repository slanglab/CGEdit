# CGEdit/

Models and training data to accompany the paper: Masis, Neal, Green, and O'Connor. ["Corpus-Guided Contrast Sets for Morphosyntactic Feature Detection in Low-Resource English Varieties."](https://aclanthology.org/2022.fieldmatters-1.2/) Field Matters Workshop at COLING, 2022.

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
  - `preprocessCORAAL.py`: code used to preprocess CORAAL transcript files for extrinsic evaluation (see Section 6 in the paper); note that only interviewee speech files were used for our evaluation, not any interviewer speech files
  - Note that these scripts may require modifications in order to run on your computer
  
  
## Training models

Run the train script with the contrast set generation method ('CGEdit' or 'CGEdit-ManualGen') as the first argument and the language ('AAE' or 'IndE') as the second argument. For example: 
  
    python train.py CGEdit-ManualGen AAE 

Please contact us if you would like access to our fine-tuned models for research use.


## Evaluation

The eval script will print a prediction in [0, 1] for each linguistic feature, for each test example.

Run the eval script with the contrast set generation method used for training ('CGEdit' or 'CGEdit-ManualGen') as the first argument, the language ('AAE' or 'IndE') as the second argument, and the test set filename as the third argument (not included in this repo). For example:

    python eval.py CGEdit-ManualGen AAE testFileName
