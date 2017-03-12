API_STAGE = "development"

#import warnings
#warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import json
import boto3
from StringIO import StringIO
import datetime as dt
import tempfile

region_bucket_map = {"us-east-1":      "slicematrixio",
                     "us-west-1":      "slicematrixiouswest",
                     "eu-central-1":   "slicematrixioeurope1",
                     "ap-southeast-1": "slicematrixioasia1"}

region_api_map    = {"us-east-1":      "ud7p0wre43",
                     "us-west-1":      "4u0pr4ljei",
                     "eu-central-1":   "pozdfzsgae",
                     "ap-southeast-1": "gbq29whx4l"}

class ConnectIO():
  def __init__(self, api_key, region = "us-east-1"):
    self.api_key  = api_key
    self.region   = region
    self.bucket   = region_bucket_map[region]
    self.api      = region_api_map[region]
    self.uploader = Uploader(api_key, self.region, self.bucket, self.api)

  def put_df(self, name, dataframe):
    self.uploader.put_df(name, dataframe)

  def list_files(self):
    return self.uploader.list_files()  

  def create_pipeline(self, name, type, params = {}):
    # make post request to pipeline/create
    #print("creating the pipeline!")
    #print(name)
    url = 'https://' + self.api + '.execute-api.' + self.region + '.amazonaws.com/' + API_STAGE + '/pipelines/create'
    #print url
    headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}
    body    = {'name': name, 'type': type, 'params': params}
    r = requests.post(url, verify = False, headers = headers, data=json.dumps(body))
    return json.loads(r.text)
  
  def run_pipeline(self, name, model, type = None, dataset = None, matrix_name = None, matrix_type = None, X = None, Y = None, extra_params = {}, memory = "large"):
    url = 'https://' + self.api + '.execute-api.' + self.region + '.amazonaws.com/' + API_STAGE + '/pipelines/run'
    headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}
    body = {'name': name, 'model': model, 'memory': memory, 'type': type}
    if dataset != None:
        body['dataset']     = dataset
    else:
        body['dataset']     = ""
    if matrix_name != None and matrix_type != None:
        body['matrix_name'] = matrix_name
        body['matrix_type'] = matrix_type
    if X != None and Y != None:
        body['X'] = X
        body['Y'] = Y
    for param_key in extra_params.keys():
      body[param_key] = extra_params[param_key]
    #print body
    r = requests.post(url, verify = False, headers = headers, data=json.dumps(body))
    return json.loads(r.text)

  def call_model(self, model, type, method, extra_params = {}, memory = "large"):
    url = 'https://' + self.api + '.execute-api.' + self.region + '.amazonaws.com/' + API_STAGE + '/models/call'
    headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}
    body    = {'model': model, 'type': type, 'function': method, 'extra_params': extra_params, 'memory': memory}
    r = requests.post(url, verify = False, headers = headers, data=json.dumps(body))
    #print r.text
    return json.loads(r.text)#[method]

  #def delete_pipeline(self, name, type):

  #def delete_model(self, model, type):
  
class Uploader():
  def __init__(self, api_key, region, bucket, api):
    self.api_key = api_key
    self.region  = region
    self.bucket  = bucket
    self.api     = api
	
  def get_upload_url(self, file_name):
    url = 'https://' + self.api + '.execute-api.' + self.region + '.amazonaws.com/' + API_STAGE + '/datasets/authorize'
    headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}
    r = requests.post(url, verify = False, headers = headers, data = json.dumps({'name':file_name}))
    return json.loads(r.text)

  def put_df(self, name, df):
    post = self.get_upload_url(name)
    files = {"file": df.to_csv()}
    response = requests.post(post["url"], data=post["fields"], files=files)
    return response
	
  def list_files(self): 
    url = 'https://' + self.api + '.execute-api.' + self.region + '.amazonaws.com/' + API_STAGE + '/datasets'
    headers = {'x-api-key': self.api_key}
    r = requests.get(url, verify = False, headers = headers)
    return json.loads(r.text)

