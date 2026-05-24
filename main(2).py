#updatation of main.py with more accuracy and no of dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

# Download and load SpaCy model
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')

# Function to preprocess text using SpaCy
def preprocess_text(text):
  doc = nlp(text.lower())
  tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
  return ' '.join(tokens)

# Define sample data
def get_sample_data():
  texts = [
    "The transplantation was successful with no major complications and the patient is recovering well.",
    "The transplantation failed due to a severe infection.",
    "Recovery was slower than expected, but the transplantation was ultimately successful.",
    "The patient rejected the transplanted organ shortly after the procedure.",
    "The transplantation was a complete success, with all initial tests showing positive results.",
    "There were complications during the procedure, but the outcome was still deemed successful.",
    "The transplantation did not take, and the patient experienced multiple issues post-operation.",
    "The procedure went smoothly and the patient is showing good signs of recovery.",
    "Despite some initial issues, the transplantation was considered a success in the long run.",
    "The patient experienced rejection of the transplanted organ, leading to an unsuccessful outcome.",
    "The transplant surgery was a resounding success, and the patient is expected to make a full recovery.",
    "Unfortunately, the transplantation failed due to organ rejection by the patient's body.",
    "While the transplantation itself went well, the patient's recovery has been slower than anticipated.",
    "The patient's body accepted the transplanted organ, and the procedure was deemed a complete success.",
    "Complications arose during the transplantation, but the medical team was able to resolve them successfully.",
    "The transplanted organ did not function properly, leading to an unsuccessful outcome for the patient.",
    "The transplantation was a success, and the patient is already showing signs of improved health.",
    "Despite the best efforts of the medical team, the transplantation was unsuccessful due to unforeseen complications.",
    "The patient's recovery after the successful transplantation has been remarkable and exceeding expectations.",
    "The transplanted organ was rejected by the patient's body, necessitating further medical intervention."
  ]
  labels = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0]
  return pd.DataFrame({'text': texts, 'success': labels})

# Use sample data
data = get_sample_data()

# Preprocess text data
data['text'] = data['text'].apply(preprocess_text)

# Split data into features and target
X = data['text']
y = data['success']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tokenize and build vocab
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
  for text in data_iter:
    yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(X), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def text_pipeline(x):
  return vocab(tokenizer(x))

# Pad sequences
def pad_batch(batch, padding_idx):
  text_list, labels = zip(*batch)
  text_list = [torch.tensor(text) for text in text_list]
  labels = torch.tensor(labels)
  padded_texts = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=padding_idx)
  return padded_texts, labels

# Create DataLoader
train_texts = [text_pipeline(text) for text in X_train]
test_texts = [text_pipeline(text) for text in X_test]

train_data = list(zip(train_texts, y_train))
test_data = list(zip(test_texts, y_test))

train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=lambda batch: pad_batch(batch, vocab["<unk>"]))
test_loader = DataLoader(test_data, batch_size=2, shuffle=False, collate_fn=lambda batch: pad_batch(batch, vocab["<unk>"]))

# Define model
class TextClassificationModel(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    super(TextClassificationModel, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    x = self.embedding(x)
    x, _ = self.rnn(x)
    x = self.fc(x[:, -1, :])
    x = self.softmax(x)
    return x

# Initialize model, loss function, and optimizer
model = TextClassificationModel(len(vocab), embedding_dim=100, hidden_dim=128, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
  for epoch in range(num_epochs):
    model.train()
    for texts, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(texts)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

train_model(model, train_loader, criterion, optimizer)

# Evaluate model
def evaluate_model(model, test_loader):
  model.eval()
  y_true, y_pred_prob = [], []
   
  with torch.no_grad():
    for texts, labels in test_loader:
      outputs = model(texts)
      y_true.extend(labels.numpy())
      y_pred_prob.extend(outputs[:, 1].numpy()) # Extracting probability for class 1

  y_true = np.array(y_true)
  y_pred_prob = np.array(y_pred_prob)

  accuracy = accuracy_score(y_true, np.round(y_pred_prob))
  precision = precision_score(y_true, np.round(y_pred_prob), zero_division=0)
  recall = recall_score(y_true, np.round(y_pred_prob), zero_division=0)
  f1 = f1_score(y_true, np.round(y_pred_prob), zero_division=0)

  # Calculate AUC value
  fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
  auc_value = auc(fpr, tpr)

  print(f"Accuracy: {accuracy:.2f}")
  print(f"Precision: {precision:.2f}")
  print(f"Recall: {recall:.2f}")
  print(f"F1-score: {f1:.2f}")
  print(f"AUC: {auc_value:.2f}")
  print("Confusion Matrix:\n", confusion_matrix(y_true, np.round(y_pred_prob)))
  print("Classification Report:\n", classification_report(y_true, np.round(y_pred_prob), zero_division=0))

  # ROC Curve
  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_value)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.0])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC Curve')
  plt.legend(loc="lower right")
  plt.show()
   
  return accuracy, precision, recall, f1, auc_value

accuracy, precision, recall, f1, auc_value = evaluate_model(model, test_loader)

# Predict success rate percentages for sample input
def predict_success(model, text):
  model.eval()
  with torch.no_grad():
    tokenized_text = text_pipeline(text)
    tokenized_text = torch.tensor(tokenized_text).unsqueeze(0)
    output = model(tokenized_text)
    success_prob = output[0][1].item() # Probability of class 1
    return success_prob

# Example prediction with sample data
sample_texts = [
  "The transplantation was successful with no major complications and the patient is recovering well.",
  "The transplantation failed due to a severe infection.",
  "Recovery was slower than expected, but the transplantation was ultimately successful.",
  "The patient rejected the transplanted organ shortly after the procedure.",
  "The transplantation was a complete success, with all initial tests showing positive results.",
  "There were complications during the procedure, but the outcome was still deemed successful.",
  "The transplantation did not take, and the patient experienced multiple issues post-operation.",
  "The procedure went smoothly and the patient is showing good signs of recovery.",
  "Despite some initial issues, the transplantation was considered a success in the long run.",
  "The patient experienced rejection of the transplanted organ, leading to an unsuccessful outcome.",
  "The transplant surgery was a resounding success, and the patient is expected to make a full recovery.",
  "Unfortunately, the transplantation failed due to organ rejection by the patient's body.",
  "While the transplantation itself went well, the patient's recovery has been slower than anticipated.",
  "The patient's body accepted the transplanted organ, and the procedure was deemed a complete success.",
  "Complications arose during the transplantation, but the medical team was able to resolve them successfully.",
  "The transplanted organ did not function properly, leading to an unsuccessful outcome for the patient.",
  "The transplantation was a success, and the patient is already showing signs of improved health.",
  "Despite the best efforts of the medical team, the transplantation was unsuccessful due to unforeseen complications.",
  "The patient's recovery after the successful transplantation has been remarkable and exceeding expectations.",
  "The transplanted organ was rejected by the patient's body, necessitating further medical intervention."
]

# Predicting success probabilities for all sample texts
for text in sample_texts:
  success_prob = predict_success(model, text)
  print(f"Text: {text}")
  print(f"Predicted Success Probability: {success_prob:.2f}\n")