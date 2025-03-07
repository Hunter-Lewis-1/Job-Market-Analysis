from collections import Counter

def extract_topics(text, top_n=10):
    """Extracts top words as pseudo-topics based on frequency."""
    words = text.split()
    word_counts = Counter(words)
    return word_counts.most_common(top_n)
