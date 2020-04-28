from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity

from ir_crosslingual.embeddings import embeddings
from ir_crosslingual.sentences import sentences
from ir_crosslingual.supervised_classification import sup_model
from ir_crosslingual.utils import strings

from ir_crosslingual.features import text_based
from ir_crosslingual.features import vector_based

from ir_crosslingual.main import app


def init_word_embeddings():
    # TODO: Retrieve languages from sup_binary.html and initialize all languages from this list
    german = embeddings.WordEmbeddings('de')
    german.load_embeddings()

    english = embeddings.WordEmbeddings('en')
    english.load_embeddings()

    embeddings.WordEmbeddings.learn_projection_matrix(src_lang='en', trg_lang='de')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/supervised_binary')
def sup_binary():
    return render_template('sup_binary.html')


@app.route('/supervised_binary/predict', methods=['POST'])
def sup_predict():
    if request.method == 'POST':
        src_sentence = request.form['src_sen']
        trg_sentence = request.form['trg_sen']
        src_language = request.form['src_lan']
        trg_language = request.form['trg_lan']

        if src_language == trg_language:
            return render_template('result_binary.html', prediction=-4.1, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())

        if src_sentence == "":
            return render_template('result_binary.html', prediction=-1.1, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())
        if trg_sentence == "":
            return render_template('result_binary.html', prediction=-1.2, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())

        if request.form.get('rb_sup_model') == 'rb_log_reg':
            print('Logistic Regression chosen for evaluation')
            name = 'logReg_v0.2'
        elif request.form.get('rb_sup_model') == 'rb_lstm':
            print('LSTM chosen for evaluation')
            name = 'lstm'
            return render_template('result_binary.html', prediction=-3.1, model=name.upper(),
                                   src_sentence=src_sentence, src_language=src_language.capitalize(),
                                   trg_sentence=trg_sentence, trg_language=trg_language.capitalize())

    model, prepared_features, features = sup_model.SupModel.load_model(name=name)

    source = embeddings.WordEmbeddings.get_embeddings(language=strings.languages_inversed[src_language])
    target = embeddings.WordEmbeddings.get_embeddings(language=strings.languages_inversed[trg_language])
    sens = sentences.Sentences(src_words=source, trg_words=target)
    data = sens.load_data(src_sentences=src_sentence, trg_sentences=trg_sentence, single_source=True, features=prepared_features)

    try:
        if data == -1:
            return render_template('result_binary.html', prediction=-2.1, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())
        if data == -2:
            return render_template('result_binary.html', prediction=-2.2, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())
    except ValueError:
        pass

    data = sens.extract_features(features_dict=features, data='all')
    prediction = model.predict(np.asarray(data[[feature for values in features.values()
                                                for feature in values]]).reshape(1, -1))

    return render_template('result_binary.html', prediction=prediction, src_sentence=src_sentence, trg_sentence=trg_sentence,
                           src_language=src_language.capitalize(), trg_language=trg_language.capitalize())


@app.route('/supervised_l2r')
def sup_l2r():
    return render_template('sup_l2r.html')


@app.route('/supervised_l2r/rank', methods=['POST'])
def sup_rank():
    if request.method == 'POST':
        src_sentence = request.form['src_sen']
        trg_sentence = request.form['trg_sen'].split('\n')
        src_language = request.form['src_lan']
        trg_language = request.form['trg_lan']
        k = request.form.get('k')

        europarl_check = request.form.get('europarl_check')

        if src_language == trg_language:
            return render_template('result_l2r.html', prediction=-4.1, src_sentence=src_sentence, trg_sentence='\n'.join(trg_sentence),
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())

        if src_sentence == "":
            return render_template('result_l2r.html', prediction=-1.1, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())

        if europarl_check:
            print('Loading europarl data for target sentences')
            trg_sentence = None
        elif (not europarl_check) and (trg_sentence[0] == ''):
            return render_template('result_l2r.html', prediction=-1.2, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())

        if request.form.get('rb_sup_model') == 'rb_log_reg':
            print('Logistic Regression chosen for evaluation')
            name = 'logReg_v0.2'
        elif request.form.get('rb_sup_model') == 'rb_lstm':
            print('LSTM chosen for evaluation')
            name = 'lstm'
            return render_template('result_l2r.html', prediction=-3.1, model=name.upper(),
                                   src_sentence=src_sentence, src_language=src_language.capitalize())

    model, prepared_features, features = sup_model.SupModel.load_model(name=name)

    source = embeddings.WordEmbeddings.get_embeddings(language=strings.languages_inversed[src_language])
    target = embeddings.WordEmbeddings.get_embeddings(language=strings.languages_inversed[trg_language])
    sens = sentences.Sentences(src_words=source, trg_words=target)
    sens.load_data(src_sentences=src_sentence, trg_sentences=trg_sentence, single_source=True, features=prepared_features)
    sens.data.drop_duplicates(subset='trg_sentence', keep='first', inplace=True)
    sens.extract_features(features_dict=features, data='all')

    top_sens, top_probs = sup_model.SupModel.rank_trg_sentences(model, sens, single_source=True, evaluation=False)

    if top_probs[0] < 0.5:
        return render_template('result_l2r.html', prediction=0, src_sentence=src_sentence, trg_sentence=trg_sentence,
                               src_language=src_language.capitalize(), trg_language=trg_language.capitalize())

    top_sens = top_sens if k == 'all' else top_sens[:int(k)]
    top_probs = top_probs if k == 'all' else top_probs[:int(k)]

    # TODO: Stop loop if probability is < 0.5
    # top_sens = top_sens if k == 'all' else [top_sens[i] for i in range(k) if top_probs[i] >= 0.5]
    # top_probs = top_probs if k == 'all' else [top_probs[i] for i in range(k) if top_probs[i] >= 0.5]

    return render_template('result_l2r.html', prediction=1, src_sentence=src_sentence,
                           num_sentences=len(top_sens), top_sens=top_sens, top_probs=['%.2f' % (i * 100) for i in top_probs],
                           src_language=src_language.capitalize(), trg_language=trg_language.capitalize())


if __name__ == '__main__':
    init_word_embeddings()
    app.run(debug=True)
