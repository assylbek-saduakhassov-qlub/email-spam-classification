from collections import defaultdict

def create_vocabulary(filename, thresh=40):
    word_count = defaultdict(int)

    with open(filename, 'r') as file:
        for line in file:
            words = line.split()
            for word in words:
                word_count[word] += 1

    vocabulary = [word for word, count in word_count.items() if count >= thresh]
    
    return vocabulary

def create_feature_vectors(filename, vocabulary):
    x = []
    y = []

    with open(filename, 'r') as file:
        for line in file:
            words = line.split()

            label = int(words[0])
            y.append(1 if label == 1 else -1)

            feature_vector = [0] * len(vocabulary)
            email_words = set(words[1:])
            for i, word in enumerate(vocabulary):
                if word in email_words:
                    feature_vector[i] = 1

            x.append(feature_vector)
    
    return x, y
