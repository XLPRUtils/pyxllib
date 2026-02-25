import re

def natural_sort_key(s):
    """
    Sort strings with embedded numbers naturally.
    """
    if not isinstance(s, str):
        return [s]
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def natural_sort(l): 
    """ 
    Sort the given list in the way that humans expect. 
    """ 
    l.sort(key=natural_sort_key) 
    return l
