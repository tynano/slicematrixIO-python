API_STAGE = "development"

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import requests
import json
import boto3
from StringIO import StringIO
import datetime as dt
import tempfile

class ConnectIO():
  def __init__(self, api_key):
    self.api_key  = api_key
    self.uploader = Uploader(api_key)

  def put_file(self, name, fpath):
    self.uploader.put_file(name, fpath)

  def put_df(self, name, dataframe):
    self.uploader.put_df(name, dataframe)

  def delete_file(self, key):
    self.uploader.delete_file(key)

  def list_files(self):
    return self.uploader.list_files()  

  def create_pipeline(self, name, type, params = {}):
    # make post request to pipeline/create
    url = 'https://ud7p0wre43.execute-api.us-east-1.amazonaws.com/' + API_STAGE + '/pipelines/create'
    headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}
    body    = {'name': name, 'type': type, 'params': params}
    r = requests.post(url, verify = False, headers = headers, data=json.dumps(body))
    return json.loads(r.text)
  
  def run_pipeline(self, name, model, type = None, dataset = None, matrix_name = None, matrix_type = None, X = None, Y = None, extra_params = {}, memory = "large"):
    url = 'https://ud7p0wre43.execute-api.us-east-1.amazonaws.com/' + API_STAGE + '/pipelines/run'
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
    url = 'https://ud7p0wre43.execute-api.us-east-1.amazonaws.com/' + API_STAGE + '/models/call'
    headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}
    body    = {'model': model, 'type': type, 'function': method, 'extra_params': extra_params, 'memory': memory}
    r = requests.post(url, verify = False, headers = headers, data=json.dumps(body))
    #print r.text
    return json.loads(r.text)#[method]

  #def delete_pipeline(self, name, type):

  #def delete_model(self, model, type):
  
class Uploader():
  def __init__(self, api_key):
    self.api_key = api_key
    self.credentials = None

  def check_creds(self):
    if self.credentials == None:
      return False
    else:
      # check if expiration is still good
      now = dt.datetime.now()
      delta = self.credentials['Expiration'] - now
      if delta.total_seconds() <= 0:
        return False
      else:
        return True

  def allocate(self):
    url = 'https://ud7p0wre43.execute-api.us-east-1.amazonaws.com/' + API_STAGE + '/datasets/authorize'
    headers = {'x-api-key': self.api_key, 'Content-Type': 'application/json'}
    r = requests.post(url, verify = False, headers = headers)
    self.credentials = json.loads(r.text)
    self.credentials['Expiration'] = pd.to_datetime(self.credentials['Expiration'])
    self.client = boto3.client('s3',
      aws_access_key_id=self.credentials['AccessKeyId'],
      aws_secret_access_key=self.credentials['SecretAccessKey'],
      aws_session_token=self.credentials['SessionToken'],
    )

  def put_file(self, name, fpath):
    if self.check_creds() == False:
      self.allocate()
    return self.client.upload_file(fpath, 'slicematrixio', "uploads/" + self.api_key + "/" + name)

  def put_df(self, name, dataframe):
    if self.check_creds() == False:
      self.allocate()
    tf = tempfile.NamedTemporaryFile()
    dataframe.to_csv(tf.name)
    response = self.client.upload_file(tf.name, 'slicematrixio', "uploads/" + self.api_key + "/" + name)
    tf.close()
    return response

  def delete_file(self, key):
    if self.check_creds() == False:
      self.allocate()
    key = "uploads/" + self.api_key + "/" + key
    return self.client.delete_object(Bucket='slicematrixio', Key = key)
  
  def list_files(self): 
    url = 'https://ud7p0wre43.execute-api.us-east-1.amazonaws.com/' + API_STAGE + '/datasets'
    headers = {'x-api-key': self.api_key}
    r = requests.get(url, verify = False, headers = headers)
    return json.loads(r.text)

