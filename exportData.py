import json
import csv
import dask.dataframe as dd
import os

softSkills = []
technicalSkills = []

with open("data/predicted_data.csv", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row["category"] == "soft":
            softSkills.append(row["word"])
        else:
            technicalSkills.append(row["word"])

with open("data/updated_jsonData.jsonl", "w", encoding="utf-8") as tempFile:
    with open("data/data.json", encoding="utf-8") as jsonFile:
        for row in jsonFile:
            jsonData = json.loads(row)
            jsonData.update({"soft_skills":{},
                            "technical_skills":{}})
            
            for soft in softSkills:
                if jsonData["description"]["text"].count(" " + soft + " ") != 0:
                    jsonData["soft_skills"].update({soft:jsonData["description"]["text"].count(" " + soft + " ")})

            for technical in technicalSkills:
                if jsonData["description"]["text"].count(" " + technical + " ") != 0:
                    jsonData["technical_skills"].update({technical:jsonData["description"]["text"].count(" " + technical + " ")})

            try:
                jsonData["occupation_field"] = jsonData["occupation_field"]["label"]
                jsonData["occupation_group"] = jsonData["occupation_group"]["label"]
                jsonData["occupation"] = jsonData["occupation"]["label"]
                jsonData["salary_type"] = jsonData["salary_type"]["label"]
                jsonData["working_hours_type"] = jsonData["working_hours_type"]["label"]
                jsonData["description"] = jsonData["description"]["text"]

                jsonData.pop("external_id")
                jsonData.pop("webpage_url")
                jsonData.pop("logo_url")

                json.dump(jsonData, tempFile, ensure_ascii=False)
                tempFile.write("\n")
            except:
                pass


"""
os.remove("data/data.json")
os.remove("data/predicted_data.csv")
"""