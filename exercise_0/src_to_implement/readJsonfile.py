
import json


def readJsonfile(file):
    # Opening JSON file
    f = open(file)
    # returns JSON object as a dictionary
    data = json.load(f)
    f.close()
    return data


jdata = readJsonfile('Labels.json')
