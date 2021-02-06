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

    fs = Grid.GridPartitioner(data=train, npart=10)
    model = chen.ConventionalFTS(partitioner=fs)
    model.fit(train)
    # print(model)
    forecasts = model.predict(test)

    data = {'message': 'Hello world', 'forecast': forecasts}

    return JsonResponse(data)
