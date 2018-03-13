df_text=df2.loc[:,'text']
n_row=df_text.shape[0]
for i in range(n_row):
    df_text[i]=clean_data(df_text[i])


df_list = [df_l.split() for df_l in df_text.values]

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(df_list)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in df_list]

Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=10, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=10, num_words=10))



