# Import the collections package for the circular list
import collections

# Import the numpy package for mean and std
import numpy as np

NoneType = type(None);

class CircularList:

    def __init__(self, maxlen, default_value = None):

        self.data = collections.deque(maxlen=maxlen);
        self.last_element = None;

        if not isinstance(default_value, NoneType):
            for i in range(maxlen):
                self.data.append(default_value);
                self.last_element = default_value;

    def append(self, i):
        self.data.append(i);
        self.last_element = i;

    def values(self):
        return list(self.data);

    def elements(self):
        return list(self.data);

    def mean(self):
        return np.mean(self.data);

    def std(self):
        return np.std(self.data);

    def len(self):
        return len(self.data);
