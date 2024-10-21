<h1>Natural Language Processing with Disaster Tweets</h1>

<p>This project is a solution to the Kaggle competition <a href="https://www.kaggle.com/competitions/nlp-getting-started">Natural Language Processing with Disaster Tweets</a>. The goal is to build a machine learning model that can predict whether a given tweet is about a real disaster (<code>1</code>) or not (<code>0</code>).</p>

<h2>Table of Contents</h2>

<ul>
  <li><a href="#overview">Overview</a></li>
  <li><a href="#data-loading">Data Loading</a></li>
  <li><a href="#data-preprocessing">Data Preprocessing</a>
    <ul>
      <li><a href="#correcting-target-errors">Correcting Target Errors</a></li>
      <li><a href="#text-cleaning">Text Cleaning</a></li>
      <li><a href="#handling-abbreviations">Handling Abbreviations</a></li>
    </ul>
  </li>
  <li><a href="#model-building">Model Building</a>
    <ul>
      <li><a href="#tokenization">Tokenization</a></li>
      <li><a href="#custom-dataset-class">Custom Dataset Class</a></li>
      <li><a href="#model-initialization">Model Initialization</a></li>
    </ul>
  </li>
  <li><a href="#training-and-evaluation">Training and Evaluation</a>
    <ul>
      <li><a href="#training-arguments">Training Arguments</a></li>
      <li><a href="#metrics-computation">Metrics Computation</a></li>
      <li><a href="#training-execution">Training Execution</a></li>
      <li><a href="#evaluation">Evaluation</a></li>
    </ul>
  </li>
  <li><a href="#prediction-and-submission">Prediction and Submission</a></li>
  <li><a href="#conclusion">Conclusion</a></li>
  <li><a href="#acknowledgments">Acknowledgments</a></li>
</ul>

<h2 id="overview">Overview</h2>

<p>In this project, we:</p>

<ul>
  <li>Load and preprocess the dataset.</li>
  <li>Correct mislabeled data points.</li>
  <li>Perform extensive text cleaning and preprocessing.</li>
  <li>Use a pretrained Transformer model (DeBERTa) for sequence classification.</li>
  <li>Fine-tune the model on the training data.</li>
  <li>Evaluate the model on a validation set.</li>
  <li>Make predictions on the test set.</li>
  <li>Prepare a submission file for Kaggle.</li>
</ul>

<h2 id="data-loading">Data Loading</h2>

<pre><code># Data Loading
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train["text"][:50])
print(train.columns)
print(test.columns)
</code></pre>

<p>We begin by loading the training and test datasets provided by the competition and inspecting the first few entries and columns.</p>

<h2 id="data-preprocessing">Data Preprocessing</h2>

<h3 id="correcting-target-errors">Correcting Target Errors</h3>

<p>Based on <a href="https://www.kaggle.com/code/wrrosa/keras-bert-using-tfhub-modified-train-data#About-this-kernel">wrrosa's Kaggle notebook</a>, certain tweets are incorrectly labeled. We correct these mislabeled data points:</p>

<pre><code># Correcting the target values
ids_with_target_error = [328, 443, 513, 2619, 3640, 3900, 4342, 5781,
                         6552, 6554, 6570, 6701, 6702, 6729, 6861, 7226]
train.loc[train['Unnamed: 0'].isin(ids_with_target_error), 'target'] = 0
</code></pre>

<h3 id="text-cleaning">Text Cleaning</h3>

<p>We perform several preprocessing steps ("Remove Emojis" is inspired by <a href="https://www.kaggle.com/code/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert#5.-Data-Cleaning-">vbmokin's Kaggle notebook)</a>:</p>

<ul>
  <li><strong>Remove Emojis</strong>: Using regex patterns.</li>
  <li><strong>Remove URLs</strong>: Eliminating links starting with <code>http</code> or <code>www</code>.</li>
  <li><strong>Remove Mentions and Hashtags</strong>: Stripping out Twitter handles and hashtags.</li>
  <li><strong>Remove Numbers</strong>: Deleting all numeric characters.</li>
  <li><strong>Lowercasing</strong>: Converting text to lowercase.</li>
  <li><strong>Remove Punctuation</strong>: Eliminating punctuation marks.</li>
  <li><strong>Remove Stopwords</strong>: Removing common English stopwords using NLTK.</li>
  <li><strong>Replace Abbreviations</strong>: Expanding abbreviations to their full forms.</li>
  <li><strong>Remove Repeated Characters</strong>: Condensing repeated characters (e.g., "loooove" → "love").</li>
</ul>

<pre><code># Text Cleaning Functions
import os
import re
import string
import nltk
from nltk.corpus import stopwords

# Set up NLTK data directory
nltk_data_dir = '/home/stefan/nltk_data'
os.environ['NLTK_DATA'] = nltk_data_dir
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
nltk.download('omw-1.4', download_dir=nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess(text):
    text = remove_emoji(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    for word in text.split():
        if word.lower() in abbreviations:
            text = text.replace(word, abbreviations[word.lower()])
    text = re.sub(r'(.)\1{1,}', r'\1', text)
    return text
</code></pre>

<h3 id="handling-abbreviations">Handling Abbreviations</h3>

<p>We expand common internet abbreviations based on a dictionary inspired by <a href="https://www.kaggle.com/rftexas/text-only-kfold-bert">rftexas's Kaggle notebook</a>:</p>

<pre><code># Abbreviations Dictionary
abbreviations = {
    "$": " dollar ",
    "€": " euro ",
    "4ao": "for adults only",
    "a.m": "before midday",
    # ... (additional abbreviations)
    "zzz": "sleeping bored and tired"
}
</code></pre>

<p>We apply preprocessing to the relevant columns:</p>

<pre><code>
columns_to_preprocess = ['text', 'keyword', 'location']
for column in columns_to_preprocess:
    train[column] = train[column].fillna('Missing')
    test[column] = test[column].fillna('Missing')

train['clean_text'] = train['text'].apply(preprocess)
test['clean_text'] = test['text'].apply(preprocess)
train['clean_keyword'] = train['keyword'].apply(preprocess)
test['clean_keyword'] = test['keyword'].apply(preprocess)
train['clean_location'] = train['location'].apply(preprocess)
test['clean_location'] = test['location'].apply(preprocess)

train.drop(columns=['text', 'keyword', 'location', 'Unnamed: 0'], inplace=True)
test.drop(columns=['text', 'keyword', 'location', 'id'], inplace=True)

# Save cleaned data
train.to_csv('train_cleaned.csv', index=False)

train.drop(columns=['clean_keyword', 'clean_location'], inplace=True)
test.drop(columns=['clean_keyword', 'clean_location'], inplace=True)
</code></pre>

<h2 id="model-building">Model Building</h2>

<h3 id="tokenization">Tokenization</h3>

<p>We use the <code>DebertaTokenizer</code> from Hugging Face to tokenize the text data:</p>

<pre><code>from transformers import DebertaTokenizer
from sklearn.model_selection import train_test_split

tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train['clean_text'].tolist(),
    train['target'].tolist(),
    test_size=0.2,
    random_state=42
)
  
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test['clean_text'].tolist(), truncation=True, padding=True)
</code></pre>

<h3 id="custom-dataset-class">Custom Dataset Class</h3>

<p>We create a custom <code>Dataset</code> class to handle our tokenized data:</p>

<pre><code>import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, encodings, labels=None): 
        self.encodings = encodings
        self.labels = labels  

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)
test_dataset = CustomDataset(test_encodings)
</code></pre>

<h3 id="model-initialization">Model Initialization</h3>

<p>We initialize the DeBERTa model for sequence classification:</p>

<pre><code>from transformers import DebertaForSequenceClassification

model = DebertaForSequenceClassification.from_pretrained('microsoft/deberta-base')
</code></pre>

<h2 id="training-and-evaluation">Training and Evaluation</h2>

<h3 id="training-arguments">Training Arguments</h3>

<p>We set up the training arguments for the <code>Trainer</code>:</p>

<pre><code>from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=1,              
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=64,   
    warmup_steps=100,               
    weight_decay=0.01,               
    logging_dir='./logs',            
    logging_steps=50,                
    evaluation_strategy='steps',     
    eval_steps=1000,                
    save_steps=2000,                 
    load_best_model_at_end=True,     
    metric_for_best_model='f1',      
    greater_is_better=True,          
)
</code></pre>

<h3 id="metrics-computation">Metrics Computation</h3>

<p>We define a function to compute evaluation metrics during training:</p>

<pre><code>def compute_metrics(p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}
</code></pre>

<h3 id="training-execution">Training Execution</h3>

<p>We use the <code>Trainer</code> class from Hugging Face to train the model:</p>

<pre><code>from transformers import Trainer

trainer = Trainer(
    model=model,                        
    args=training_args,                  
    train_dataset=train_dataset,        
    eval_dataset=val_dataset,           
    compute_metrics=compute_metrics      
)

trainer.train()
</code></pre>

<h3 id="evaluation">Evaluation</h3>

<p>After training, we evaluate the model on the validation set:</p>

<pre><code># Evaluate the model
val_results = trainer.evaluate()
print(val_results)
</code></pre>

<h2 id="prediction-and-submission">Prediction and Submission</h2>

<p>We make predictions on the test set and prepare the submission file:</p>

<pre><code># Make predictions
test_predictions = trainer.predict(test_dataset)
preds = test_predictions.predictions.argmax(-1)

# Prepare submission
submission = pd.read_csv('sample_submission.csv')
submission['target'] = preds
submission.to_csv('submission_deberta.csv', index=False)
</code></pre>

<h2 id="conclusion">Conclusion</h2>

<p>In this project, we built a text classification model to distinguish between disaster-related and non-disaster tweets. By leveraging a pretrained DeBERTa model and performing thorough text preprocessing inspired by the Kaggle community, we aimed to improve the model's ability to understand and classify the nuances in the tweets.</p>

<p><strong>Next Steps:</strong></p>

<ul>
  <li>Experiment with different pretrained models like BERT or RoBERTa.</li>
  <li>Perform hyperparameter tuning to improve model performance.</li>
  <li>Implement cross-validation for a more robust evaluation.</li>
</ul>

<h2 id="acknowledgments">Acknowledgments</h2>

<p>We would like to thank the Kaggle community members whose work inspired parts of this project:</p>

<ul>
  <li><a href="https://www.kaggle.com/code/wrrosa/keras-bert-using-tfhub-modified-train-data#About-this-kernel">wrrosa's notebook on Keras BERT using TFHub</a> for insights on correcting target errors.</li>
  <li><a href="https://www.kaggle.com/code/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert#5.-Data-Cleaning-">vbmokin's notebook on NLP EDA and data cleaning</a> for text preprocessing techniques.</li>
  <li><a href="https://www.kaggle.com/rftexas/text-only-kfold-bert">rftexas's notebook on Text-Only KFold BERT</a> for handling abbreviations in text data.</li>
</ul>

<p><strong>References:</strong></p>

<ul>
  <li><a href="https://www.kaggle.com/competitions/nlp-getting-started">Kaggle Competition: Natural Language Processing with Disaster Tweets</a></li>
  <li><a href="https://huggingface.co/transformers/">Hugging Face Transformers Documentation</a></li>
  <li><a href="https://github.com/microsoft/DeBERTa">DeBERTa: Decoding-enhanced BERT with Disentangled Attention</a></li>
</ul>


