# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#imports
import numpy as np
import pandas as pd
#machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
subm = pd.read_csv('../input/sample_submission.csv')

def train_model(label_cols):
            clf=MultinomialNB()
            for i in label_cols:
                clf.fit(X_train_tfidf, train[i].values)
                subm[i] = clf.predict_proba(X_test_tfidf)
            return subm
def  main():
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        train['none'] = 1-train[label_cols].max(axis=1)
        
        
        
        train['comment_text'].fillna("unknown", inplace=True)
        test['comment_text'].fillna("unknown", inplace=True)
        
        
        import re, string
        re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
        def tokenize(s): return re_tok.sub(r' \1 ', s).split()
        
        
        
        n = train.shape[0]
        vec = CountVectorizer(ngram_range=(1,3), tokenizer=tokenize, max_features=150000)
        trn_term_doc = vec.fit_transform(train['comment_text'].values)
        test_term_doc = vec.transform(test['comment_text'])
        
        
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(trn_term_doc)
        X_train_tfidf.shape
        
        
        X_test_tfidf = tfidf_transformer.transform(test_term_doc)
        X_test_tfidf.shape
        
        from sklearn.naive_bayes import MultinomialNB
        label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        clf=MultinomialNB()
        train_model(label_cols)
        
        subm.to_csv("submission_test.csv", index=False)
        
if __name__=='__main__':
    main()