import numpy as np

class HMM_Model:
    def __init__(self, tags, words):
        self.tags = list(tags)
        self.words = list(words)
        self.tags_dict = {tags[i] : i for i in range(len(tags))}
        self.words_dict = {words[i] : i for i in range(len(words))}

        number_of_tags = len(tags)
        number_of_words = len(words)

        self.t = np.zeros((number_of_tags, number_of_tags))
        self.e = np.zeros((number_of_tags, number_of_words))
        self.q = np.zeros((number_of_tags, 1))

    def mle (self, x , y):
        """
        Calculate the maximum likelihood estimators for the transition and
        emission distributions , in the multinomial HMM case .
        : param x : an iterable over sequences of POS tags
        : param y: a matching iterable over sequences of words
        : return : a tuple (t , e ) , with
        t . shape = (| val ( X )| ,| val ( X )|) , and
        e . shape = (| val ( X )| ,| val ( Y )|)
        """

        # emission table

        for index in range(len(x)):
            for t, w in zip(x[index], y[index]):
                self.e[self.tags_dict[t]][self.words_dict[w]] += 1

        # transition table
        for index in range(len(x)):
            for t1, t2 in zip(x[index][:-1], x[index][1:]):
                self.t[self.tags_dict[t1]][self.tags_dict[t2]] += 1

        #calculate q
        for POS in x:
            self.q[self.tags_dict[POS[0]]] += 1

        self.t = (self.t.T / self.t.sum(axis=1)).T
        self.e = (self.e.T / self.e.sum(axis=1)).T
        self.q = self.q / self.q.sum()
        return (self.t, self.e, self.q)


    def sample (self, Ns , xvals , yvals):
        """
        sample sequences from the model .
        : param Ns : a vector with desired sample lengths , a sample is generated per
        entry in the vector , with corresponding length .
        : param xvals : the possible values for x variables , ordered as in t , and e
        : param yvals : the possible values for y variables , ordered as in e
        : param t : the transition distributions of the model
        : param e : the emission distributions of the model
        : return : x , y - two iterables describing the sampled sequences .
        """

        np.random.choice(self.tags, len(Ns), p=self.q)
        x = np.zeros(len(Ns))
        y = np.zeros(len(Ns))
        for index, sentence_len in enumerate(Ns):
            curr_x = np.zeros(sentence_len)
            curr_y = np.zeros(sentence_len)
            curr_x[0] = np.random.choice(self.tags, 1, p=self.q)
            curr_y[0] = np.random.choice(self.words, 1, p=self.e[self.tags_dict[curr_x[0]], :])

            for word_loc in range(1, sentence_len):
                curr_x[word_loc] = np.random.choice(self.tags, 1, p=self.t[self.tags_dict[curr_x[word_loc - 1]], :])
                curr_y[word_loc] = np.random.choice(self.words, 1, p=self.e[self.tags_dict[curr_x[word_loc]], :])

            x[index] = curr_x
            y[index] = curr_y

        return x, y

    def viterbi (self, y , suppx):
        """
        Calculate the maximum a - posteriori assignment of x ’s .
        : param y : a sequence of words
        : param suppx : the support of x ( what values it can attain )
        : param t : the transition distributions of the model
        : param e : the emission distributions of the model
        : return : xhat , the most likely sequence of hidden states ( parts of speech ).
        """
        n = len(y)
        num_tags = len(self.tags)

        prob_v = np.zeros(n, num_tags)
        track_v = np.zeros(n, num_tags)

        prob_v[0, :] = self.q
        for i in range(1, n):
            for j in range(num_tags):
                temp = np.multiply(np.multiply(prob_v[i -1, :], self.t[:, j]), self.e[:, self.words_dict[y[i]]])
                prob_v[i, j ] = np.max(temp)
                track_v[i, j] = np.argmax(temp)

        # calc route
        route = np.nan(n)
        route[n] = self.tags_dict[np.argmax(prob_v[n,:])]
        #track = track_v[np.max(prob_v[n,:])]
        for i in range(n - 1, 0, -1):
            route[i] = self.tags_dict[track_v[self.tags_dict[route[i+1]]]]

        return route

