import torch
import json
import random

from config import DefaultConfig
from model import Net
from nltk_utils import tokenize, stem, bag_of_words

with open('intents.json','r') as json_data:
    intents = json.load(json_data)

opt = DefaultConfig()
file = 'checkpoints/model_0430_22:00:32.pth'

data = torch.load(file)
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = Net(opt.input_size, opt.hidden_size, opt.output_size)
model.load_state_dict(model_state)

# evaluation
model.eval()

bot_name = 'Victoria'
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input('You: ')
    if sentence == 'quit':
        print('End of Conversation')
        break

    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    out = model(x)
    _, predict = torch.max(out, dim =1)
    tag = tags[predict.item()]

    probs = torch.softmax(out, dim = 1)
    print(probs)
    prob = probs[0][predict.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I don't understand...")






