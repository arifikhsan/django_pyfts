from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from pyFTS.data import Enrollments
from pyFTS.partitioners import Grid
from pyFTS.models import chen, cheng
from rest_framework import status
from rest_framework.response import Response

import json
import warnings


def index(request):
    warnings.filterwarnings('ignore')

    train = Enrollments.get_data()
    test = train

    return __fts(train, test)

@csrf_exempt
@require_POST
def dynamic(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)

    train = body['train']
    test = body['test']

    # print(test)
    # print(request.body['train'])

    # data = {'message': 'Aaaa'}
    # return JsonResponse(data)

    return __fts(train, test)

@csrf_exempt
@require_POST
def model_cheng(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)

    train = body['train']
    test = body['test']

    return __fts(train, test, 'cheng')

def __fts(train, test, model_type='chen'):
    df = Enrollments.get_dataframe()
    # aa = df['Enrollments'].values.tolist()

    fs = Grid.GridPartitioner(data=train, npart=10)
    if model_type == 'chen':
        model = chen.ConventionalFTS(partitioner=fs)
    elif model_type == 'cheng':
        model = cheng.TrendWeightedFTS(partitioner=Grid.GridPartitioner(data=train, npart=7))
    else:
        model = chen.ConventionalFTS(partitioner=fs)

    model.fit(train)
    forecasts = model.predict(test)

    # data = {'message': 'Hello world' }
    data = {'train': train, 'test': test, 'forecast': forecasts}
    return JsonResponse(data)
