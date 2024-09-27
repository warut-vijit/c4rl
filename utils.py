from wonderwords import RandomWord

def get_random_name() -> str:
    random_word_generator = RandomWord()
    random_adjective = random_word_generator.word(include_parts_of_speech=["adjectives"])
    random_noun = random_word_generator.word(include_parts_of_speech=["nouns"])
    return f"{random_adjective}_{random_noun}"
