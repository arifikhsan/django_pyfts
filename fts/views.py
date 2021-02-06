from django.shortcuts import render
from django.http import JsonResponse

from pyFTS.data import Enrollments
from pyFTS.partitioners import Grid
from pyFTS.models import chen
import warnings
import numpy

def index(request):
    warnings.filterwarnings('ignore')

    train = Enrollments.get_data()
    test = Enrollments.get_data()
    df = Enrollments.get_dataframe()

    data = df['Enrollments'].values
    # print(df['Enrollments'].values)
    # universe of discosure partitioner
    partitioner = Grid.GridPartitioner(data=train, npart=10)
    # create an empty model using the Chen(1996) method
    model = chen.ConventionalFTS(partitioner=partitioner)
    # the training procedure is performed by the method fit
    model.fit(train)
    # model.fit(data)
    # print the model rules
    # print(model)
    # the forecasting procedure is performed bu the method predict
    # forecasts = model.predict(test)

    # print(Enrollments.get_dataframe())
    # print(pyFTS)
    data = {'message': 'Hello world'}
    return JsonResponse(data)
