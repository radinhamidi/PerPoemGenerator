import numpy as np
from keras.models import model_from_json
import sys
from dataGenerator import *
import nltk
import random

# load json and create model
json_file = open('./charLSTM.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("./charLSTM.h5")
print("Loaded model from disk")

text = getCharText()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 15
charNumb = -1
input  = 'عش است از آسمان'

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

for diversity in [0.2, 0.5, 1.0, 1.2]:
    print('----- diversity:', diversity)

    generated = ''
    generated += input
    print('----- Generating with seed: "' + input + '"')
    sys.stdout.write(generated)

    for i in range(charNumb):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(input):
            x_pred[0, t, char_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = input[1:] + next_char

        if next_char=='\n' and charNumb==-1:
            break

        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()



start_index = random.randint(0, len(text) - maxlen - 1)
sentence = text[start_index: start_index + maxlen]
reference = text[start_index: start_index + maxlen + charNumb]

generated = ''
generated += sentence
print('----- Generating with seed: "' + sentence + '"')
sys.stdout.write(generated)

for i in range(charNumb):
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        try:
            x_pred[0, t, char_indices[char]] = 1.
        except:
            pass

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
print()

print(nltk.translate.bleu_score.corpus_bleu(reference, generated))