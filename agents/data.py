class Data:
    __data__ = {'states': [],
                'actions': [],
                'log_probs': [],
                'values': [],
                'entropies': [],
                'rewards': [],
                'masks': []}

    def add(self, **kwargs):
        for key, value in kwargs.items():
            self.__data__[key].append(value)
    
    def get(self, key):
        return self.__data__[key]
    
    def clear(self):
        self.__data__['states'].clear()
        self.__data__['actions'].clear()
        self.__data__['log_probs'].clear()
        self.__data__['values'].clear()
        self.__data__['entropies'].clear()
        self.__data__['rewards'].clear()
        self.__data__['masks'].clear()