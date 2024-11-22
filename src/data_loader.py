nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

!wget --continue --tries=3 http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
file = open('glove.6B.100d.txt', 'r', encoding='utf-8')
content = file.readlines()
file.close()
embeddings = {}

for line in content:
    split_line = line.split()
    word = split_line[0]
    embedding = np.array([float(val) for val in split_line[1:]])
    embeddings[word] = embedding

<data load from kaggle>
# load  everything from files
import pickle
with open('word2index.pickle', 'rb') as handle:
    word2index = pickle.load(handle)
embedding_matrix = np.load('embedding_metrix.npy')
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
Xtrain = np.load('Xtrain.npy')
Xtest = np.load('Xtest.npy')
Ytrain = np.load('Ytrain.npy')
Ytest = np.load('Ytest.npy')
# model = load_model('model.h5')
history = np.load('history.npy', allow_pickle='TRUE').item()
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)


X = train_data['text']
Y = train_data['label']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
word2index = tokenizer.word_index

def get_maxlen(data):
    maxlen = 0
    for sent in data:
        maxlen = max(maxlen, len(sent))
    return maxlen
Xtokens = tokenizer.texts_to_sequences(X)
maxlen = get_maxlen(Xtokens)
print(maxlen)

Xtrain = pad_sequences(Xtokens, maxlen = maxlen,  padding = 'post', truncating = 'post')
Xtrain.shape
Ytrain = to_categorical(Y)
Ytrain.shape

Ytrain = to_categorical(Y)
embed_size = 100

<train test split >

embedding_matrix = np.zeros((len(word2index) + 1, embed_size))

for word, i in word2index.items():
    if word in embeddings:
        embedding_matrix[i] = embeddings[word]
    else:
        print(word)
        embedding_matrix[i] = np.random.normal(0, 1, embed_size)




lemmatizer=WordNetLemmatizer()

def preprocessing(text):
    # Ensure the input is a string, otherwise return an empty string
    if not isinstance(text, str):
        return ''

    # Use regex to match words starting with @ or #
    # remove URLs

    cleaned_text = re.sub(r'(http|https|www)\S+', '', text)  # Remove URLs
    cleaned_text = re.sub(r'[@#]\w+', '', cleaned_text) # Remove mentions (like @username) and hashtgs
    # cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)

    # remove every charactor which is not alphbet
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    cleaned_text = word_tokenize(cleaned_text)
    filtered_words = [lemmatizer.lemmatize(word, pos='v') for word in cleaned_text]

    # remove word if it's not in embeddings
    # new_filtered_words = [word for word in filtered_words if word in embeddings]
    # if len(new_filtered_words) != len(filtered_words):
    #     return None

    # Join the words back into a single string
    text = ' '.join(filtered_words)
    return text

def get_maxlen(data):
    maxlen = 0
    for sent in data:
        maxlen = max(maxlen, len(sent))
    return maxlen



word2index = tokenizer.word_index

embed_size = 100
maxlen = 41



