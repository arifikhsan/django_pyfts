from django.shortcuts import render
from django.http import JsonResponse

from pyFTS.data import Enrollments
from pyFTS.partitioners import Grid
from pyFTS.models import chen
import warnings


def index(request):
    warnings.filterwarnings('ignore')

    train = Enrollments.get_data()
    test = Enrollments.get_data()
    df = Enrollments.get_dataframe()
    aa = df['Enrollments'].values.tolist()

    fs = Grid.GridPartitioner(data=train, npart=10)
    model = chen.ConventionalFTS(partitioner=fs)
    model.fit(train)
    # print(aa)
    # forecasts = model.predict(test)
    forecasts = model.predict([18876])

    # data = {'message': 'Hello world', 'train': aa }
    data = {'message': 'Hello world', 'train': aa, 'forecast': forecasts}

    return JsonResponse(data)
