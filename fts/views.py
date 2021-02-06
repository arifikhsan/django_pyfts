from django.shortcuts import render
from django.http import JsonResponse
# from pyFTS.data import Enrollments

def index(request):
    # print(Enrollments.get_dataframe())
    # print(pyFTS)
    data = {'message': 'Hello world'}
    return JsonResponse(data)
