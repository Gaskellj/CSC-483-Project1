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

def create_test_models(c):
    afghanistanModel = create_ngram_model_lines(NgramModelWithInterpolation,"train/af.txt",c)
    print('done one')
    chinaModel = create_ngram_model_lines(NgramModelWithInterpolation,"train/cn.txt",c)
    print('done one')
    germanyModel = create_ngram_model_lines(NgramModelWithInterpolation,"train/de.txt",c)
    print('done one')
    finlandModel = create_ngram_model_lines(NgramModelWithInterpolation,"train/fi.txt",c)
    print('done one')
    franceModel = create_ngram_model_lines(NgramModelWithInterpolation,"train/fr.txt",c)
    print('done one')
    indiaModel = create_ngram_model_lines(NgramModelWithInterpolation,"train/in.txt",c)
    print('done one')
    irelandModel = create_ngram_model_lines(NgramModelWithInterpolation,"train/ir.txt",c)
    print('done one')
    pakistanModel = create_ngram_model_lines(NgramModelWithInterpolation,"train/pk.txt",c)
    print('done one')
    southAfricaModel = create_ngram_model_lines(NgramModelWithInterpolation,"train/za.txt",c)
    print("created dictionaries")
    models = [afghanistanModel,chinaModel,germanyModel,finlandModel,franceModel,indiaModel,irelandModel,pakistanModel,southAfricaModel]

    return models

def return_strings_file(path, stringArray):
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            stringArray.append(line)
    return stringArray

def test_models_with_validation(modelsArray, validation_file_path, c):
    strArray = []
    return_strings_file(validation_file_path,strArray)
    model_number = 0
    bestProbability = 0
    for model in modelsArray:
        totalNgrams = 0
        model_number +=1
        totalProbability = 0
        for line in strArray:
            Ngrams = ngrams(c,line)
            for Ngram in Ngrams:
                totalNgrams += 1
                totalProbability += (model.prob((Ngram[0]),(Ngram[1])))
        
        avgProbability = totalProbability / totalNgrams
        if avgProbability > bestProbability:
            bestProbability = avgProbability
            country = COUNTRY_CODES[model_number-1]
        #print("Average probability for " + COUNTRY_CODES[model_number-1] + " is " + str(avgProbability) + "\n\n")
    print("The most likely country is: " + country)

def test_models_with_validation2(modelsArray, validation_file_path, c):
    strArray = []
    return_strings_file(validation_file_path, strArray)
    model_number = 0
    best_probability = 0
    for line in strArray:
        Ngrams = ngrams(c,line)
        for ngram in Ngrams:
            model_count = 0
            for model in modelsArray:
                current_prob = 0
                current_prob += (model.prob((ngram[0]),(ngram[1])))
            print(COUNTRY_CODES[model_count] + "  " + line + "  "+str(current_prob))
            model_count += 1
    pass

def test_single_city(modelsArray, Ngrams, c):
    model_count = 0
    best_probability = 0
    for model in modelsArray:
        model
        total_probability = 0
        for ngram in Ngrams:
            total_probability += (model.prob((ngram[0]),(ngram[1])))
        avg_probability = total_probability/len(Ngrams)
        #print("probability for " + COUNTRY_CODES[model_count] + " is: " + str(avg_probability))
        if avg_probability > best_probability:
            best_probability = avg_probability
            country = COUNTRY_CODES[model_count]
        model_count+=1
    return country

def find_accuracy(country_code,model_context, modelsArray, filepath):
    strArray = []
    correct_selection = 0
    total_count = 0
    return_strings_file(filepath,strArray)
    for City in strArray:
        CityNgrams = ngrams(model_context,City)
        country_returned = test_single_city(modelsArray, CityNgrams, model_context)
        if country_returned == country_code:
            correct_selection +=1
        total_count +=1
    return correct_selection / total_count


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
    
    def prob(self, context, char):
        ContextCount = 0
        NgramCount = 0
        prob = 0
        if(context,char) in self.NgramDict:
            NgramCount = (self.NgramDict[context,char])
            ContextCount = self.context_dictionary[context]
        if (context) not in self.context_dictionary:
            if self.k > 0:
                prob = (self.k/(self.k *len(self.vocabDict)))
            else:
                prob = (1/len(self.vocabDict))
        elif (context) in self.context_dictionary and NgramCount == 0:
            if self.k > 0:
                prob = self.k/(self.context_dictionary[context] + (self.k * len(self.vocabDict))) 
        else:
            if self.k > 0:
                prob = (NgramCount + self.k) / (ContextCount + (self.k * len(self.vocabDict)))      
            else:
                prob = (NgramCount / ContextCount)                                           
        return prob

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
            string+=(self.random_char(string[-self.context:]))
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
        self.lambdas = []
        pass

    def get_vocab(self):
        super().get_vocab()
        pass

    def update_new(self,text):
        context = self.context
        for i in range(0,context):
            super().update(context,text)

    def update(self, text):
        Ngrams = []
        context = 0
        while context <= self.context:
            Ngrams += ngrams(context,text)
            for n in Ngrams:
                if len(n[0]) == context:
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
            context +=1
        #print (self.context_dictionary)
        #print(self.NgramDict)
        self.set_default_lambdas()

    def prob(self, context, char):
        probability = 0
        if self.context == 0:
            probability = super().prob("",char)
        else:
            for i in range(0,len(context)+1):
                current_context = context[i:]
                probability += (self.lambdas[i] * (super().prob(current_context,char)))
        return probability

    def change_lambdas(self):
        print("\nEntered values must sum to 1\n")
        for i in range(0,self.context):
            Lambda = int(input("Please enter value for lamda "+ str(i+1)+ ": "))
            self.lambdas.append(Lambda)
        print("Lambdas populated")
    
    def set_default_lambdas(self):
        for i in range(0,self.context+1):
            self.lambdas.append(1/(self.context+1))

################################################################################
# Your N-Gram Model Experimentations
################################################################################

# Add all code you need for testing your language model as you are
# developing it as well as your code for running your experiments
# here.
#
# Hint: it may be useful to encapsulate it into multiple functions so
# that you can easily run any test or experiment at any time.

m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 3)
print(m.perplexity("From fairest creatures we desire increase, That thereby beauty's rose might never die,"))
print(m.perplexity("Mr. Golden exemplifies, perhaps in a cautionary way, how easy it has become to gamble on"))


"""
#m = NgramModel(1,0)
#m.update('abab')
#m.update('abcd')
#random.seed(1)
#print(m.perplexity('abcd'))
#print(m.perplexity('abca'))
#print(m.perplexity('abcda'))
#print(m.random_text(25))
#print([m.random_char('d') for i in range(25)])

#m = create_ngram_model(NgramModel, 'shakespeare_input.txt', 7)
#print('model_created')
#print(m.random_text(250))


#print(m.prob('d','a'))

#m = NgramModelWithInterpolation(2, 1)
#m.update('abab')
#m.update('abcd')
#print(m.prob('~a','b'))
#print(m.prob('ba','b'))
#print(m.prob('~c','d'))
#print(m.prob('bc','d'))
#m.change_lambdas()
#random.seed(1)
#print(m.prob('a','a'))
#print(m.prob('a','b'))

model_context = 2

models = create_test_models(model_context)

#test_models_with_validation2(models, "val/af.txt",model_context)
#test_models_with_validation2(models, "val/cn.txt",model_context)
#test_models_with_validation2(models, "val/de.txt",model_context)
#test_models_with_validation2(models, "val/fi.txt",model_context)
#test_models_with_validation2(models, "val/fr.txt",model_context)
#test_models_with_validation2(models, "val/in.txt",model_context)
#test_models_with_validation2(models, "val/ir.txt",model_context)
#test_models_with_validation2(models, "val/pk.txt",model_context)
#test_models_with_validation2(models, "val/za.txt",model_context)

#while True:
    #City = input("Please enter a city name: ")
    #CityNgrams = ngrams(model_context,City)
    #print(City + " is probably in " + test_single_city(models,CityNgrams,model_context))
    #pass


af_accuracy = find_accuracy('af',model_context,models,"val/af.txt")
print("The accuracy of the af model is: " +str(af_accuracy))
cn_accuracy = find_accuracy('cn',model_context,models,"val/cn.txt")
print("The accuracy of the cn model is: " +str(cn_accuracy))
de_accuracy = find_accuracy('de',model_context,models,"val/de.txt")
print("The accuracy of the de model is: " +str(de_accuracy))
fi_accuracy = find_accuracy('fi',model_context,models,"val/fi.txt")
print("The accuracy of the fi model is: " +str(fi_accuracy))
fr_accuracy = find_accuracy('fr',model_context,models,"val/fr.txt")
print("The accuracy of the fr model is: " +str(fr_accuracy))
in_accuracy = find_accuracy('in',model_context,models,"val/in.txt")
print("The accuracy of the in model is: " +str(in_accuracy))
ir_accuracy = find_accuracy('ir',model_context,models,"val/ir.txt")
print("The accuracy of the ir model is: " +str(ir_accuracy))
pk_accuracy = find_accuracy('pk',model_context,models,"val/pk.txt")
print("The accuracy of the pk model is: " +str(pk_accuracy))
za_accuracy = find_accuracy('za',model_context,models,"val/za.txt")
print("The accuracy of the za model is: " +str(za_accuracy))



strArray = []
return_strings_file("val/af.txt",strArray)
totalProbability = 0
for line in strArray:
    Ngrams = ngrams(2,line)
    for Ngram in Ngrams:
        #print(Ngram)
        totalProbability += (afghanistanModel.prob((Ngram[0]),(Ngram[1])))
avgProbability = totalProbability / len(Ngrams)
print("afghan" + str(avgProbability))

strArray = []
return_strings_file("val/af.txt",strArray)
totalProbability = 0
for line in strArray:
    Ngrams = ngrams(2,line)
    for Ngram in Ngrams:
        #print(Ngram)
        totalProbability += (chinaModel.prob((Ngram[0]),(Ngram[1])))
avgProbability = totalProbability / len(Ngrams)
print("china" + str(avgProbability))

"""





