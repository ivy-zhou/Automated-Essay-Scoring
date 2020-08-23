from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse

from .forms import AnswerForm

from .utils.model import *
from .utils.helpers import *

from .models import Essay

import numpy as np
import os
import pickle
current_path = os.path.abspath(os.path.dirname(__file__))

# Create your views here.


def index(request):
    essay_list = Essay.objects.all()
    context = {
        'essay_list': essay_list,
    }
    return render(request, 'grader/index.html', context)


def essay(request, essay_id):
    essay = get_object_or_404(Essay, pk=essay_id)
    context = {
        "essay": essay,
    }
    return render(request, 'grader/essay.html', context)


def submit(request):
    # Handle post
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = AnswerForm(request.POST)
        if form.is_valid():

            content = form.cleaned_data.get('answer')

            # Calculate stats
            wc = word_count(content)
            spelling_and_grammar = accuracy(content)
            tense_percentage = tense(content)
            semantic = semantic_score(content)

            if len(content) > 20:
                num_features = 200

                # load glove embeddings
                embedding_dict = {}
                with open(os.path.join(
                        current_path, "deep_learning_files/embeddings.pickle"), 'rb') as handle:
                    embedding_dict = pickle.load(handle)

                model = embedding_dict

                clean_test_essays = []
                clean_test_essays.append(essay_to_wordlist(
                    content, remove_stopwords=True))
                testDataVecs = getAvgFeatureVecs(
                    clean_test_essays, model, num_features)
                testDataVecs = np.array(testDataVecs)
                testDataVecs = np.reshape(
                    testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

                lstm_model = get_model()
                lstm_model.load_weights(os.path.join(
                    current_path, "deep_learning_files/final_lstm.h5"))
                preds = lstm_model.predict(testDataVecs)

                if math.isnan(preds):
                    preds = 0
                else:
                    preds = np.round(preds)

                if preds < 0:
                    preds = 0
            else:
                preds = 0

            K.clear_session()
            essay = Essay.objects.create(
                content=content,
                score=preds,
                semantic=semantic,
                tense=tense_percentage,
                accuracy=spelling_and_grammar,
                wordcount=wc
            )
        return redirect('essay', essay_id=essay.id)
    else:
        form = AnswerForm()

    context = {
        "form": form,
    }
    return render(request, 'grader/submit.html', context)
