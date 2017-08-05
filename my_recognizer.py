import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
   
    for word_index in range(0,test_set.num_items):
        
        score_dict={}
        X, lengths= test_set.get_all_Xlengths()[word_index]
        best_score=float("-inf")
        predicted_word=None
        for word, HMMtrained_model in models.items():
            try:
                score_on_test_word=HMMtrained_model.score(X, lengths)
                score_dict[word]=score_on_test_word
                if score_on_test_word>best_score:
                    best_score=score_on_test_word
                    predicted_word=word
            except: pass
        probabilities.append(score_dict)    
        guesses.append(predicted_word)    
    
    return probabilities, guesses
