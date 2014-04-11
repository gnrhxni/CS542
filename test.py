
import nettalk_data

def test_dictionary_len():
    entries_iterator = nettalk_data.dictionary()
    entries_list = list(entries_iterator)
    assert len(entries_list) == 20007

def test_top1000_len():
    top1000 = nettalk_data.topK
    assert len(top1000) == 1000

def test_dictionary_attrs():
    """ Check to make sure that an example dictionary entry has its attributes
    """
    entry = nettalk_data.dictionary().next()
    assert hasattr(entry, 'word')
    assert hasattr(entry, 'phonemes')
    assert hasattr(entry, 'stress')
    assert hasattr(entry, 'flag')

def test_sane_dictionary():
    """ Check that no dictionary entries have missing elements
    """
    for entry in nettalk_data.dictionary():
        assert len(entry) == 4
        assert all( len(field) > 0 for field in entry )

def test_wordstream():
    stream = nettalk_data.wordstream()
    assert stream.next() == ['---aard', '--aardv', '-aardva', 
                             'aardvar', 'ardvark', 'rdvark-', 
                             'dvark--', 'vark---']

def test_wordstream_windowsize():
    stream = nettalk_data.wordstream(windowsize=6)
    assert stream.next() == ['---aar', '--aard', '-aardv', 
                             'aardva', 'ardvar', 'rdvark',
                             'dvark-', 'vark--']

def test_wordstream_length():
    stream = nettalk_data.wordstream()
    assert len(list(stream)) == 20007


def test_wordstream_custom_entries():
    class entry_looking_thing(object):
        def __init__(self, word):
            self.word = word

    custom_entries = [
        entry_looking_thing(word)
        for word in ('aardvark', 'aback', 'abacus', 'anathema')
    ]

    stream = nettalk_data.wordstream(input_entries=custom_entries)
    for entry, wordsalad in zip(custom_entries, stream):
        assert len(entry.word) == len(wordsalad)


def test_articFeatureNames_length():
    # not sure that 29 is right
    assert len(nettalk_data.articFeatureNames) == 29

def test_stressFeatureNames_length():
    assert len(nettalk_data.stressFeatureNames) == 5

def test_stressFeatures_length():
    assert len(nettalk_data.stressFeatures) == 6

