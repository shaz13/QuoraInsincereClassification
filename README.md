# QuoraInsincereClassification

Top 3% solution for Quora Insincere Classification 

![](https://www.socialsamosa.com/socialketchup/wp-content/uploads/2017/11/Bollywood-characters-on-Quora.jpg)



This was a kernel competition and we couldn't use external data or leverage pre-trained models. And, the runtime limit was just 2 hours. 

## Preprocessing
The preprocessing plays a crucial role in NLP competition. Although network architecture plays an important role, with good processing, one can come up with divergent models in less time. The following methods worked pretty well
- Cleaning punctuation, special characters and removing non-English and filling NaN - "__##__". Discussion on why not 0s, "na" instead of  "__##__"  (Let's brainstorm this intuition at Slack)
- Not removing stopwords. We lose much information in doing so at this competition
- Adding Out of vocabulary words in embeddings
- Replacing mostly misspelled words after EDA

## Model Architecture
We used a simple GRU and LSTM network written in Keras with the custom tokenizer. Also, we used Attention Is all you need paper ideas and added Attention layer to our model with several layers
- Embeddings + 
- Attention Layer
- Max Pooling, Average Pooling --> Concat with static features (length, words, unique word ratio, punctuations and other features from raw text)

## Cross-Validation
We use simple KFold split and validated our models offline. As this was kernels only competition we compressed a lot of our tasks to fit into the runtime. The final submission was generated from an ensemble of 5 models with respective weights. 

The code for the 97th solution as well as the best single model will be soon open-sourced. This was fun. And, in fact, my first International NLP competition. Lot's of learning forward. Have a great weekend!


(To Be Updated with EDA



### Team 
- Mohsin Hassan
- Mohammad Shahebaz
- Ajay 
- Nickil Maveli
