from flask import Flask, request, render_template_string
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
import io
import base64
import random

app = Flask(__name__)

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
        "The transplantation maybe failed due to a severe infection.",
        "Recovery was slower than expected, but the transplantation was ultimately successful.",
        "The patient rejected the transplanted organ shortly after the procedure.",
        "The transplantation was a complete success, with all initial tests showing positive results.",
        "There were complications during the procedure, but the outcome was still deemed successful."
    ]
    labels = [1, 0, 1, 0, 1, 1]
    return pd.DataFrame({'text': texts, 'success': labels})

# Initialize and preprocess data
data = get_sample_data()
data['text'] = data['text'].apply(preprocess_text)

# Function to randomly select 6 samples from the data
def get_random_samples(data, n=6):
    indices = random.sample(range(len(data)), n)
    return data.iloc[indices]

# Split data into features and target
X = data['text']
y = data['success']
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
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        x = self.sigmoid(x)
        return x

# Initialize model, loss function, and optimizer
model = TextClassificationModel(len(vocab), embedding_dim=100, hidden_dim=128, output_dim=1)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

train_model(model, train_loader, criterion, optimizer)

# Evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred_prob = [], []

    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts).squeeze()
            y_true.extend(labels.numpy())
            y_pred_prob.extend(outputs.numpy())

    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)

    accuracy = accuracy_score(y_true, np.round(y_pred_prob))
    precision = precision_score(y_true, np.round(y_pred_prob), zero_division=0)
    recall = recall_score(y_true, np.round(y_pred_prob), zero_division=0)
    f1 = f1_score(y_true, np.round(y_pred_prob), zero_division=0)

    # Calculate AUC value
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc_value = auc(fpr, tpr)

    # Save ROC curve to a BytesIO object
    buf = io.BytesIO()
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(buf, format='png')
    buf.seek(0)
    roc_curve_img = base64.b64encode(buf.getvalue()).decode('utf-8')

    return accuracy, precision, recall, f1, auc_value, confusion_matrix(y_true, np.round(y_pred_prob)), classification_report(y_true, np.round(y_pred_prob), zero_division=0), roc_curve_img

# Predict success rate percentages for sample input
def predict_success(model, text):
    model.eval()
    with torch.no_grad():
        tokenized_text = text_pipeline(text)
        tokenized_text = torch.tensor(tokenized_text).unsqueeze(0)
        output = model(tokenized_text).squeeze()
        success_prob = output.item()
        return success_prob

@app.route('/')
def index():
    # Get random samples
    random_samples = get_random_samples(data)
    sample_texts = random_samples['text'].tolist()
    sample_labels = random_samples['success'].tolist()

    # Create test loader for random samples
    sample_texts = [text_pipeline(text) for text in sample_texts]
    sample_data = list(zip(sample_texts, sample_labels))
    sample_loader = DataLoader(sample_data, batch_size=2, shuffle=False, collate_fn=lambda batch: pad_batch(batch, vocab["<unk>"]))

    # Evaluate model on random samples
    accuracy, precision, recall, f1, auc_value, conf_matrix, class_report, roc_curve_img = evaluate_model(model, sample_loader)

    # Predict success probabilities for random samples
    predictions = [predict_success(model, preprocess_text(text)) for text in random_samples['text']]

    # Render HTML template with predictions and metrics
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Evaluation and Predictions</title>
        <style>
            /* Global Styles */
            body {
                font-family: 'Arial', sans-serif;
                margin: 20px;
                background-color: #f4f4f4;
                line-height: 1.6;
                color: #333;
            }
            h1, h2 {
                color: #444;
                margin-bottom: 10px;
            }
            p {
                margin-bottom: 15px;
            }
            pre {
                background-color: #f9f9f9;
                padding: 10px;
                border: 1px solid #ddd;
            }
            img {
                margin-top: 20px;
                max-width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <h1>Model Evaluation Metrics</h1>
        <p><strong>Accuracy:</strong> {{ accuracy }}</p>
        <p><strong>Precision:</strong> {{ precision }}</p>
        <p><strong>Recall:</strong> {{ recall }}</p>
        <p><strong>F1 Score:</strong> {{ f1 }}</p>
        <p><strong>AUC Value:</strong> {{ auc_value }}</p>

        <h2>Confusion Matrix</h2>
        <pre>{{ conf_matrix }}</pre>

        <h2>Classification Report</h2>
        <pre>{{ class_report }}</pre>

        <h2>ROC Curve</h2>
        <img src="data:image/png;base64,{{ roc_curve_img }}" alt="ROC Curve">

        <h2>Predictions for Random Samples</h2>
        <ul>
            {% for text, label, prediction in zip(sample_texts, sample_labels, predictions) %}
            <li><strong>Text:</strong> {{ text }} <br>
                <strong>True Label:</strong> {{ label }} <br>
                <strong>Predicted Success Probability:</strong> {{ prediction}}</li>
            {% endfor %}
        </ul>
    </body>
    </html>
    '''

    return render_template_string(html_template, accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                                  auc_value=auc_value, conf_matrix=conf_matrix, class_report=class_report,
                                  roc_curve_img=roc_curve_img, sample_texts=random_samples['text'].tolist(),
                                  sample_labels=random_samples['success'].tolist(), predictions=predictions,
                                  zip=zip)

if __name__ == '__main__':
    app.run(debug=True) 