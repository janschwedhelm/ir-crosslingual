from flask import Flask, render_template, url_for, request
import pandas as pd
import numpy as np
from sklearn.externals import joblib


from ir_crosslingual.embeddings import embeddings
from ir_crosslingual.sentences import sentences
from ir_crosslingual.supervised_classification import sup_model
from ir_crosslingual.utils import paths

from ir_crosslingual.unsupervised_classification.unsup_model import UnsupModel

from ir_crosslingual.main import app

# pre-define full list of model features
MODEL_FEATURES = ['src_sentence', 'trg_sentence', 'translation',
                  'norm_diff_translated_words', 'abs_diff_num_words', 'abs_diff_num_punctuation',
                  'abs_diff_occ_question_mark', 'abs_diff_occ_exclamation_mark',
                  'rel_diff_num_words', 'rel_diff_num_punctuation', 'norm_diff_num_words',
                  'norm_diff_num_punctuation', 'euclidean_distance', 'cosine_similarity'] \
                 + ['src_embedding_pca_{}'.format(i) for i in range(10)] \
                 + ['trg_embedding_pca_{}'.format(i) for i in range(10)]
RAW_FEATURES = ['src_sentence', 'trg_sentence']
LABEL = 'translation'

# feature order of best MLP model (average)
features_mlp = ['norm_diff_num_words', 'euclidean_distance', 'abs_diff_occ_exclamation_mark_0',
                'abs_diff_occ_question_mark_2', 'abs_diff_occ_question_mark_0', 'cosine_similarity',
                'norm_diff_translated_words', 'abs_diff_occ_exclamation_mark_1', 'abs_diff_occ_question_mark_1',
                'abs_diff_num_words', 'abs_diff_occ_exclamation_mark_2', 'abs_diff_num_punctuation','src_embedding_pca_0',
                'src_embedding_pca_1', 'src_embedding_pca_2', 'src_embedding_pca_3', 'src_embedding_pca_4',
                'src_embedding_pca_5', 'src_embedding_pca_6', 'src_embedding_pca_7', 'src_embedding_pca_8',
                'src_embedding_pca_9', 'trg_embedding_pca_0', 'trg_embedding_pca_1', 'trg_embedding_pca_2',
                'trg_embedding_pca_3', 'trg_embedding_pca_4', 'trg_embedding_pca_5', 'trg_embedding_pca_6',
                'trg_embedding_pca_7', 'trg_embedding_pca_8', 'trg_embedding_pca_9']

# load fitted models/scalers of best runs for MLP & Logistic Regression
pca = {}
mean_scaler = {}
scaler = joblib.load(open('models/scaler/ct.pkl', 'rb'))
for prefix in ['src', 'trg']:
    mean_scaler['{}'.format(prefix)] = joblib.load(open('models/mean_scaler/mean_scaler_{}.pkl'.format(prefix),
                                                        'rb'))
    pca['{}'.format(prefix)] = joblib.load(open('models/pca/pca_{}.pkl'.format(prefix), 'rb'))

mlp_model, mlp_prepared_features, mlp_features_dict = sup_model.SupModel.load_model(name='mlp_avg_best')
lr_model, lr_prepared_features, lr_features_dict = sup_model.SupModel.load_model(name='logReg_v0.2')


def init_word_embeddings():
    german = embeddings.WordEmbeddings('de')
    german.load_embeddings()

    english = embeddings.WordEmbeddings('en')
    english.load_embeddings()

    embeddings.WordEmbeddings.learn_projection_matrix(src_lang='en', trg_lang='de')


def get_transformer_feature_names(columnTransformer):

    output_features = []

    for name, pipe, features in columnTransformer.transformers_:
        if name!='remainder':
            for i in pipe:
                trans_features = []
                if hasattr(i,'categories_'):
                    trans_features.extend(i.get_feature_names(features))
                else:
                    trans_features = features
            output_features.extend(trans_features)

    return output_features


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
            name = 'log_reg_best_avg'
        elif request.form.get('rb_sup_model') == 'rb_mlp':
            print('Multilayer Perceptron chosen for evaluation')
            name = 'mlp_avg_best'

    source = embeddings.WordEmbeddings.get_embeddings(language=paths.languages_inversed[src_language])
    target = embeddings.WordEmbeddings.get_embeddings(language=paths.languages_inversed[trg_language])
    sens = sentences.Sentences(src_words=source, trg_words=target)

    if name == 'mlp_avg_best':
        data = sens.load_data(src_sentences=src_sentence, trg_sentences=trg_sentence, single_source=True,
                              features=mlp_prepared_features)
    else:
        data = sens.load_data(src_sentences=src_sentence, trg_sentences=trg_sentence, single_source=True,
                              features=lr_prepared_features)

    try:
        if data == -1:
            return render_template('result_binary.html', prediction=-2.1, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())
        if data == -2:
            return render_template('result_binary.html', prediction=-2.2, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())
    except ValueError:
        pass

    if name == 'mlp_avg_best':
        data = sens.extract_features(features_dict=mlp_features_dict, data='all')
    else:
        data = sens.extract_features(features_dict=lr_features_dict, data='all')

    if name == 'mlp_avg_best':
        # apply PCA using models fitted on training data
        for prefix in ['src', 'trg']:
            data[['{}_embedding_pca_{}'.format(prefix, i) for i in range(10)]] = \
                pd.DataFrame(pca['{}'.format(prefix)].transform(np.array(mean_scaler['{}'.format(prefix)]
                                                                .transform(np.array(data['{}_embedding'.format(prefix)])[0]
                                                                           .reshape(1, -1)))[0].reshape(1, -1)))
        data['translation'] = None
        # apply scaling using model fitted on training data
        data = pd.DataFrame(scaler.transform(data[MODEL_FEATURES]))
        data.columns = get_transformer_feature_names(scaler) + RAW_FEATURES + [LABEL]
        data = data.infer_objects()
        prediction = mlp_model.predict(data[features_mlp])
        print(mlp_model.predict_proba(data[features_mlp]))
    else:
        features = [feature for values in lr_features_dict.values() for feature in values]
        prediction = lr_model.predict(data[features])

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
            name = 'log_reg_best_avg.2'
        elif request.form.get('rb_sup_model') == 'rb_mlp':
            print('Multilayer Perceptron chosen for evaluation')
            name = 'mlp_avg_best'

    source = embeddings.WordEmbeddings.get_embeddings(language=paths.languages_inversed[src_language])
    target = embeddings.WordEmbeddings.get_embeddings(language=paths.languages_inversed[trg_language])
    sens = sentences.Sentences(src_words=source, trg_words=target)

    if name == 'mlp_avg_best':
        sens.load_data(src_sentences=src_sentence, trg_sentences=trg_sentence, single_source=True,
                       features=mlp_prepared_features)
    else:
        sens.load_data(src_sentences=src_sentence, trg_sentences=trg_sentence, single_source=True,
                       features=lr_prepared_features)

    sens.data.drop_duplicates(subset='trg_sentence', keep='first', inplace=True, ignore_index=True)

    if name == 'mlp_avg_best':
        sens.extract_features(features_dict=mlp_features_dict, data='all')
    else:
        sens.extract_features(features_dict=lr_features_dict, data='all')

    if name == 'mlp_avg_best':
        # apply PCA using models fitted on training data
        for prefix in ['src', 'trg']:

            X = np.vstack(sens.data['{}_embedding'.format(prefix)])

            sens.data[['{}_embedding_pca_{}'.format(prefix, i) for i in range(10)]] = \
                pd.DataFrame(pca['{}'.format(prefix)].transform(mean_scaler['{}'.format(prefix)].transform(X)).tolist())


        sens.data['translation'] = None
        # apply scaling using model fitted on training data
        sens.data = pd.DataFrame(scaler.transform(sens.data[MODEL_FEATURES]))
        sens.data.columns = get_transformer_feature_names(scaler) + RAW_FEATURES + [LABEL]
        sens.data = sens.data.infer_objects()

    if name == 'mlp_avg_best':
        features = features_mlp
        top_sens, top_probs = sup_model.SupModel.rank_trg_sentences(mlp_model, sens, features,
                                                                   single_source=True, evaluation=False)
    else:
        features = [feature for values in sens.features_dict.values() for feature in values]
        top_sens, top_probs = sup_model.SupModel.rank_trg_sentences(lr_model, sens, features,
                                                                    single_source=True, evaluation=False)
    if top_probs[0] < 0.5:
        return render_template('result_l2r.html', prediction=0, src_sentence=src_sentence, trg_sentence=trg_sentence,
                               src_language=src_language.capitalize(), trg_language=trg_language.capitalize())

    top_sens = top_sens if k == 'all' else top_sens[:int(k)]
    top_probs = top_probs if k == 'all' else top_probs[:int(k)]

    if name == 'mlp_avg_best':
        return render_template('result_l2r.html', prediction=1, src_sentence=src_sentence, model=name,
                               num_sentences=len(top_sens), top_sens=top_sens, top_probs=['%.4f' % (i) for i in top_probs],
                               src_language=src_language.capitalize(), trg_language=trg_language.capitalize())
    else:
        return render_template('result_l2r.html', prediction=1, src_sentence=src_sentence, model=name,
                               num_sentences=len(top_sens), top_sens=top_sens,
                               top_probs=['%.2f' % (i * 100) for i in top_probs],
                               src_language=src_language.capitalize(), trg_language=trg_language.capitalize())


@app.route('/unsupervised_binary')
def unsup_binary():
    return render_template('unsup_binary.html')


@app.route('/unsupervised_binary/predict', methods=['POST'])
def unsup_predict():
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

    model = UnsupModel()

    source = embeddings.WordEmbeddings.get_embeddings(language=paths.languages_inversed[src_language])
    target = embeddings.WordEmbeddings.get_embeddings(language=paths.languages_inversed[trg_language])
    sens = sentences.Sentences(src_words=source, trg_words=target)
    data = sens.load_data(src_sentences=src_sentence, trg_sentences=trg_sentence, single_source=True)

    try:
        if data == -1:
            return render_template('result_binary.html', prediction=-2.1, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())
        if data == -2:
            return render_template('result_binary.html', prediction=-2.2, src_sentence=src_sentence, trg_sentence=trg_sentence,
                                   src_language=src_language.capitalize(), trg_language=trg_language.capitalize())
    except ValueError:
        pass

    features = ['src_embedding_aligned', 'trg_embedding']
    prediction = model.predict_proba(data[features])[0]
    prediction = prediction[1]

    return render_template('result_unsup_binary.html', prediction=float('%.4f' % prediction), src_sentence=src_sentence, trg_sentence=trg_sentence,
                           src_language=src_language.capitalize(), trg_language=trg_language.capitalize())


if __name__ == '__main__':
    init_word_embeddings()
    app.run(debug=True)
