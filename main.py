import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import argparse


# Define the SentimentClassifier class as in the training script
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        model_name = 'bert-base-uncased'
        self.model = BertModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.model.config.hidden_size, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.drop(pooled_output)
        output = self.out(output)
        return self.softmax(output)


# Function to load the model and tokenizer
def load_model(model_path, num_classes):
    model = SentimentClassifier(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return model, tokenizer


# Function to preprocess the input sentence
def preprocess(sentence, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_len,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding['input_ids'], encoding['attention_mask']


# Function to predict sentiment
def predict(sentence, model, tokenizer, max_len):
    input_ids, attention_mask = preprocess(sentence, tokenizer, max_len)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs, dim=1)
    return prediction.item()


# Command-line interface
def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis of Financial Statements")
    parser.add_argument('sentence', type=str, help='Input sentence for sentiment analysis')
    args = parser.parse_args()

    model_path = './best.pth'
    num_classes = 3  # Change this based on the number of classes in your model
    max_len = 50

    model, tokenizer = load_model(model_path, num_classes)
    prediction = predict(args.sentence, model, tokenizer, max_len)

    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}  # Modify based on your label encoding
    print(f'Sentence: {args.sentence}')
    print(f'Predicted Sentiment: {label_mapping[prediction]}')


if __name__ == '__main__':
    main()
