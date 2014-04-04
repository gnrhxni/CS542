
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
