import requests
import json
from zipfile import ZipFile 


the_working_year = 2020

while requests.head("https://data.jobtechdev.se/annonser/historiska/" + str(the_working_year) + ".jsonl.zip").status_code == 200:
    response = requests.get("https://data.jobtechdev.se/annonser/historiska/" + str(the_working_year) + ".jsonl.zip") 

    temp = open("data.zip", 'wb')
    temp.write(response.content) 

    with ZipFile('data.zip', 'r') as zip:
        print(zip.read(str(the_working_year) + ".jsonl"))
    
    the_working_year += 1

