import os
import requests
import json
from zipfile import ZipFile 

the_working_year = 2022

while requests.head("https://data.jobtechdev.se/annonser/historiska/" + str(the_working_year) + ".jsonl.zip").status_code == 200:
    response = requests.get("https://data.jobtechdev.se/annonser/historiska/" + str(the_working_year) + ".jsonl.zip") 

    apizipfile = open("data/data.zip", "wb")
    apizipfile.write(response.content) 

    with ZipFile("data/data.zip", "r") as zip:
        unzipedfile = open("data/data.jsonl", 'wb')
        unzipedfile.write(zip.read(str(the_working_year) + ".jsonl"))
    
    with open("data/data.jsonl", 'r', encoding="utf-8") as bigFile:

        gbg = open("data/data.json", 'a', encoding="utf-8")
        for line in bigFile: 
            json_line = json.loads(line)
            try:
                if(json_line["workplace_address"]["region"] == "Västra Götalands län" and "Null" not in json_line.keys()):
                    json.dump(json_line, gbg)
                    gbg.write("\n")
            except:
                pass
            
    the_working_year += 1
    
unzipedfile.close()
os.remove("data/data.jsonl")
apizipfile.close()
os.remove("data/data.zip")
