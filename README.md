# root_health
Train a Convolutional Neural Network (CNN) using Keras to automatically classify root health without having to physically touch the plants


The actual experiment design of this project is motivated by Darrah et al. in their 2017 paper, Real- time Root Monitoring of Hydroponic Crop Plants: Proof of Concept for a New Image Analysis System.

Such a system can improve the yields of existing hydroponic farms making farms more efficient and sustainable to run. Of course, the successful application of hydroponics has massive implications for the medical marijuana industry.

The overall goal of the project was to develop an automated root growth analysis system capable of accurately measuring the roots followed by detecting any growth problems:

In particular, roots needed to be classified into two groups:

“Hairy” roots
“Non-hairy” roots
The “hairier” a root is, the better the root can suck up nutrients. The “less hairy” the root is, the fewer nutrients it can intake, potentially leading to the plant starving and dying.


Accuracy Report 
                precision    recall  f1-score   support

    hairy_root       0.68      0.86      0.76       299
non_hairy_root       0.82      0.61      0.70       311

      accuracy                           0.73       610
     macro avg       0.75      0.74      0.73       610
  weighted avg       0.75      0.73      0.73       610