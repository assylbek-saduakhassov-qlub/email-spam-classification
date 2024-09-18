def write_important_words(vocabulary, w):
    weight_word_pairs = list(zip(w, vocabulary))

    sorted_by_positive_weights = sorted(weight_word_pairs, key=lambda pair: pair[0], reverse=True)
    sorted_by_negative_weights = sorted(weight_word_pairs, key=lambda pair: pair[0])

    top_positive_words = [pair[1] for pair in sorted_by_positive_weights[:12]]
    top_negative_words = [pair[1] for pair in sorted_by_negative_weights[:12]]

    return top_positive_words, top_negative_words
