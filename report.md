# OCR assignment report
## Feature Extraction (Max 200 Words)
The train data and labels are retrieved from the files and are used to extract all the feature vectors with full dimensions. 
Then, I extracted the principal components by finding the eigenvectors of the covariance matrix of the train data. 
I reduced the dimensions by projecting the 2340 dimensions onto 40 principal components (which is a good number of axes for balancing data loss and compression). 
I reduced the dimensions of the feature vectors by choosing the best 10 principal components. The feature selection was done partially. 
I have found the best initial feature by computing the divergence between each class label and by getting the initial best feature vector with the highest divergence. 
I have also tried choosing the whole 10 set of feature vectors by using multi-divergence for multiple classes but at the end it didn’t work out, I have selected features by brute forcing. 
Nevertheless, I understand that selecting the best features would require each class compared with the other and backwards. 

## Classifier (Max 200 Words)
The method chosen for classification was the k-nearest-neighbour classification. 
To calculate the distance between feature vectors I had to choose from a range of distance measure functions from the labs and lectures and I have ended up in using the negative cosine distance as it gave out the best results. 
I had also used a number of ways of coding the classification function and most ways were taught in the lecture, but I ended up using the fastest one. 
The method finds the k nearest feature vectors and labels. 
The most frequent labels were stored as the nearest ones. 
In order to choose a sensible k, I used a function from the lecture that computed the scores for each k starting from 1 to 32 and outputs the number that the score was highest. 
Nevertheless, it didn’t display the best results, so I have used trial and error and ended up using 4.

## Error Correction (Max 200 Words)
For the error correction, I stored the dictionary in the model using a way taught to us in the lectures. 
Based on the bounding box sizes of each label, I created words from concatenating the letters and calculating the difference in the sizes 
after a space or a new line and I have also removed any unnecessary punctuation. 
I have decided that traversing the dictionary for each word will take a lot of time for evaluating the classifier, therefore, I have imported a library called ‘enchant’ and that library suggest nearest words for a word that does not exist in the dictionary. 
This library suggested updated words and take the first word out of the list of suggested words. 
The suggested word had the same length with the classified word to minimise mistakes. 
If a label was misclassified, I stored updated word and labels, and its label indices which I used later when substituting the label with the correct one in the labels array. 
After that, I have attempted to run it, but there was an error that I didn’t have time to solve therefore, I couldn’t test if it produced better or worse results.

## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
- Page 1: [97.4% correct]
- Page 2: [98.4% correct]
- Page 3: [88.6% correct]
- Page 4: [61.1% correct]
- Page 5: [44.3% correct]
- Page 6: [36.1% correct]

## Other information (Optional, Max 100 words)
I used code from the lectures and from labs. I have also found the 'enchant' library from stack overflow