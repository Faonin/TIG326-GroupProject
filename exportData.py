import json
import csv
import dask.dataframe as dd
import os
"""
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
            json.dump(jsonData, tempFile, ensure_ascii=False)
            tempFile.write("\n")
"""
ddf = dd.read_json("data/updated_jsonData.jsonl", encoding="utf-8", lines=True)

ddf.to_csv("data/output_data.csv", encoding="utf-8", single_file=True, index=True)


"""
os.remove("data/updated_jsonData.jsonl")

os.remove("data/data.json")
os.remove("data/predicted_data.csv")
"""