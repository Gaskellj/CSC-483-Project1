import math, random

# PLEASE do not delete or modify the comments that divide the code
# into sections, like the following comment.

################################################################################
# Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(c):
    ''' Returns a padding string of length c to append to the front of text
        as a pre-processing step to building n-grams. c = n-1 '''
    return '~' * c

def ngrams(c, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-c context and the second is the character '''
    ngramArray = []
    for count in range(0,len(text)):
        context = c - count
        string = ""
        tempArray = []
        if context > 0:
            string += (start_pad(context))
            for i in range (0,(c-context)):
                string += text[i]
        else:
            for i in range(c,0,-1):
                string += text[count-i]
        tempArray.append(string)
        tempArray.append(text[count])
        ngramArray.append(tempArray)

    return ngramArray

def create_ngram_model(model_class, path, c=2, k=0):
    ''' Creates and returns a new n-gram model trained on the entire text
        found in the path file '''
    model = model_class(c, k)
    print('creating')
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, c=2, k=0):
    '''Creates and returns a new n-gram model trained line by line on the
        text found in the path file. '''
    model = model_class(c, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, c, k):
        self.context = c
        self.k = k
        self.context_dictionary = {}
        self.NgramDict = {}
        self.vocabDict = []

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        for character in self.NgramDict:
            if (character[1]) not in self.vocabDict:
                self.vocabDict.append(character[1])

    def update_dictionary(self,context):
        if context in self.context_dictionary:
            self.context_dictionary[context] += 1
        else:
            self.context_dictionary[context] = 1

    def update(self, text):
        Ngrams = ngrams(self.context,text)
        for n in Ngrams:
            if tuple(n) in self.NgramDict:
                self.NgramDict[tuple(n)] += 1
            else:
                self.NgramDict[tuple(n)] = 1
            self.update_dictionary(n[0])
        myKeys = list(self.context_dictionary.keys())
        myKeys.sort()
        self.context_dictionary = {i: self.context_dictionary[i] for i in myKeys}
        self.get_vocab()
        self.vocabDict.sort()

    """
    def update_inactive(self, text):
        ''' Updates the model n-grams based on text '''
        print('doing update')
        Ngrams = ngrams(self.context,text)
        print('done Ngrams')
        for Ngram in Ngrams:
            inModel = False
            for i in range(0,len(self.Model)):
                if self.Model[i][1] == Ngram and not inModel:
                    inModel = True
                    index = i
            if inModel:
                current_value = int(self.Model[index][0])
                self.Model.append((current_value+1, Ngram))
                del self.Model[index]
            else:
                self.Model.append((1,Ngram))
        self.update_dictionary(self.Model)
        return self.Model


    def prob_inactive(self, context, char):
        ''' Returns the probability of char appearing after context '''
        found = False
        context_in_vocab = False
        for Ngram in self.Model:
            if context == Ngram[1][0]:
                context_in_vocab = True
                if char == Ngram[1][1]:
                    count = int(Ngram[0])
                    prob = count / (self.character_dictionary[Ngram[1][0]])
                    found = True
        if found:
            return prob
        elif not context_in_vocab:
            return(1/len(self.character_dictionary.keys()))
        else:
            return 0
    """
    
    def prob(self, context, char):
        ContextCount = 0
        NgramCount = 0
        prob = 0
        #self.add_k(context,char)
        if(context,char) in self.NgramDict:
            NgramCount = (self.NgramDict[context,char])
            ContextCount = self.context_dictionary[context]
        if (context) not in self.context_dictionary:
            prob = (self.k/(self.k *len(self.vocabDict)))
        elif (context) in self.context_dictionary and NgramCount == 0:
            prob = self.k/(self.context_dictionary[context] + self.k * len(self.vocabDict))
        if NgramCount != 0 and ContextCount != 0:
            prob = (NgramCount + self.k) / (ContextCount + self.k * len(self.vocabDict))
        #prob += (self.k/(len(self.vocabDict) **(self.k)))
        return prob
    
    def add_k(self, context, char):
        if (context,char) in self.NgramDict:
            self.NgramDict[context,char] += self.k
        else:
            self.NgramDict[context,char] = self.k

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        total_probability = 0
        r = random.random()
        for char in self.vocabDict:
            total_probability += self.prob(context, char)
            if total_probability > r:
                return char

    def random_text(self, length):
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        string = start_pad(self.context)
        for i in range(0,length):
            string+=(m.random_char(string[-self.context:]))
        return (string[self.context:])

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''
        total_prob = 1
        print(text)
        for i in range(self.context,len(text)):
            total_prob *= self.prob(text[i-self.context],text[i])
        try:
            Entropy = (-1/len(text)) * math.log2(total_prob)
            perplexity = 2 ** Entropy
            return perplexity
        except ValueError:
            return float('inf')

################################################################################
# N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, c, k):
        super().__init__(c,k)
        pass

    def get_vocab(self):
        super().get_vocab()
        pass

    def update(self, text):
        super().update(text)
        pass

    def prob(self, context, char):
        super().prob(context,char)
        pass

################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.

#m = NgramModel(1,1)
#m.update('abab')
#m.update('abcd')
random.seed(1)
#print(m.perplexity('abcd'))
#print(m.perplexity('abca'))
#print(m.perplexity('abcda'))
#print(m.random_text(25))
#print([m.random_char('d') for i in range(25)])

#m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 16)
#print('model_created')
#print(m.random_text(250))

#print(m.prob('a','a'))

m = NgramModelWithInterpolation(1, 0)
m.update('abab')
random.seed(1)
print(m.prob('a','a'))


