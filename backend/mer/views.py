from django.shortcuts import render
from rest_framework import viewsets
from rest_framework.response import Response
from .models import MerNN
import json
from collections import namedtuple
from rest_framework import status
from django.http.multipartparser import MultiPartParser
from rest_framework.exceptions import ParseError
from rest_framework.parsers import FileUploadParser
from django.core.files.uploadedfile import InMemoryUploadedFile
from PIL import Image
from io import BytesIO
from .mer_service import predict
from django.http import FileResponse
from django.http import HttpResponse
import base64
from io import BytesIO
from backend import urls
import cv2
import numpy as np
from .models import Latex
from .serializers import LatexSerializer
import wolframalpha
import ssl


class MERView(viewsets.ModelViewSet):

    def create(self, request, *args, **kwargs):

        # files = request.FILES['image']

        image_data = request.data['image']

        image = Image.open(BytesIO(base64.b64decode(image_data)))

        image.save('mer/assets/slika-cista.png')

        image = base64.b64decode(image_data)
        im_arr = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(im_arr, flags=cv2.IMREAD_GRAYSCALE)

        if 'rotate' in request.data and request.data['rotate'] == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        predictions = predict(image)

        # latex_object = Latex(latex_string=predictions)
        # latex_object.save()
        #
        # serializer = LatexSerializer(latex_object)

        prediction_response = {}
        prediction_response['latex_string'] = predictions

        # http: // api.wolframalpha.com / v2 / query?appid = 77KH5X-WPWXER8YQ5 & input = solve % 283 % 2
        # B1 % 29 - 2 % 5
        # Cdiv4 % 5
        # Cdiv5x % 3
        # D6 & output = json & format = plaintext

        client = wolframalpha.Client('77KH5X-WPWXER8YQ5')
        #
        res = client.query('solve '+predictions)

        # print(serializer.data)

        prediction_response['solution_pods'] = []

        print('wolfra', res)

        if res['@success'] == 'false':
            prediction_response['solution_status'] = {
                'success': False,
                'message': 'Latex string cannot be evaluated',
            }

            dm_text = ''
            if(int(res['didyoumeans']['@count'])>1):
                for _,dm in enumerate(res['didyoumeans']['didyoumean']):
                    if(dm['#text'] != 'solve'):
                        dm_text = dm_text + dm['#text'] + ' '
            prediction_response['solution_status']['didyoumean'] = dm_text

        else:
            prediction_response['solution_status'] = {
                'success': True,
                'message': 'Latex string evaluated correctly',
            }
            if 'pod' in res:
                if int(res['@numpods']) > 1:
                    for pod in res.pods:
                        solution_pod = {}
                        solution_pod['title'] = pod['@title']
                        solution_pod['subpods'] = []

                        # print(int(pod['@numsubpods']))

                        if(int(pod['@numsubpods']) > 1):

                            for sp in pod['subpod']:
                                subpod = {}
                                # print(sp)
                                if 'plaintext' in sp:
                                    subpod['plaintext'] = sp['plaintext']
                                    if 'img' in sp:
                                        subpod['img'] = {
                                            'src': sp['img']['@src'],
                                            'width': sp['img']['@width'],
                                            'height': sp['img']['@width']
                                        }
                                    solution_pod['subpods'].append(subpod)
                        else:
                            if 'subpod' in pod:
                                subpod = {}
                                if 'plaintext' in pod['subpod']:
                                    subpod['plaintext'] = pod['subpod']['plaintext']
                                    if 'img' in pod['subpod']:
                                        subpod['img'] = {
                                            'src': pod['subpod']['img']['@src'],
                                            'width': pod['subpod']['img']['@width'],
                                            'height': pod['subpod']['img']['@width']
                                        }
                                    solution_pod['subpods'].append(subpod)
                        prediction_response['solution_pods'].append(solution_pod)
                else:
                    solution_pod = {}
                    pod = res['pod']
                    solution_pod['title'] = pod['@title']
                    solution_pod['subpods'] = []
                    if (int(pod['@numsubpods']) > 1):

                        for sp in pod['subpod']:
                            subpod = {}
                            if 'plaintext' in sp:
                                subpod['plaintext'] = sp['plaintext']
                                if 'img' in sp:
                                    subpod['img'] = {
                                        'src': sp['img']['@src'],
                                        'width': sp['img']['@width'],
                                        'height': sp['img']['@width']
                                    }
                                    solution_pod['subpods'].append(subpod)
                    else:
                        if 'subpod' in pod:
                            subpod = {}
                            if 'plaintext' in pod['subpod']:
                                subpod['plaintext'] = pod['subpod']['plaintext']
                                if 'img' in pod['subpod']:
                                    subpod['img'] = {
                                        'src': pod['subpod']['img']['@src'],
                                        'width': pod['subpod']['img']['@width'],
                                        'height': pod['subpod']['img']['@width']
                                    }
                                    solution_pod['subpods'].append(subpod)
                    prediction_response['solution_pods'].append(solution_pod)

        # return Response(predictions)
        return Response(prediction_response)
