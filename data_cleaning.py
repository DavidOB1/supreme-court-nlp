import requests
import pandas as pd
import csv

data = pd.read_csv("justice.csv")
skip_nums = []

def avg(lst):
    try:
        return sum(lst) / len(lst)
    except:
        return 0

def get_ideologies(n):
    try:
        links = data[data.columns[3]].to_list()
        ideologies = []
        fullDict = requests.get(links[n]).json()
        decisionsDict = fullDict["decisions"][0]
        justiceList = decisionsDict["votes"]
        for i in range(len(justiceList)):
            ideologies.append(justiceList[i]["ideology"])
        return ideologies
    except:
        skip_nums.append(n)
        return []

def get_ideologies_list():
    output = []
    for i in range(3303):
        output.append(get_ideologies(i))
    return output

def get_issue_area_list():
    issue_areas = data[data.columns[15]].to_list()
    for i in range(len(issue_areas)):
        if type(issue_areas[i]) != str:
            issue_areas[i] = "Unknown"
    return issue_areas

def get_facts_list():
    facts = data[data.columns[8]].to_list()
    for i in range(len(facts)):
        facts[i] = facts[i][3:-5]
        facts[i].replace("â€™", "'")
    return facts

def main():
    ideologies = get_ideologies_list()
    avg_ideologies = [avg(x) for x in ideologies]
    issue_areas = get_issue_area_list()
    facts = get_facts_list()
    first_wins = data[data.columns[12]].to_list()
    first_parties = data[data.columns[6]].to_list()
    second_parties = data[data.columns[7]].to_list()
    case_names = data[data.columns[2]].to_list()
    docket_nums = data[data.columns[4]].to_list()

    with open('clean_data.csv', 'w', newline='') as csvfile:
        fieldnames = ['case_name', 'docket_num', 'first_party', 'second_party', 'facts', 
        'first_party_won', 'issue_area', 'ideologies', 'avg_ideology']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(3303):
            if not i in skip_nums:
                try:
                    writer.writerow({'case_name': case_names[i], 'docket_num': docket_nums[i],
                    'first_party': first_parties[i], 'second_party': second_parties[i], 'facts': facts[i],
                    'first_party_won': first_wins[i], 'issue_area': issue_areas[i], 'ideologies': ideologies[i],
                    'avg_ideology': avg_ideologies[i]})
                except:
                    print(f"Error on line {i}, skipping assignment.")

    print("Done")

if __name__ == "__main__":
    main()

# A few notes:
# - About 20 or so lines were skipped due to encountering errors either retrieving
#   their ideology data or in adding them to the csv file
# - Around 200 or so lines were deleted because their ideology data was wrong (all 0s)
#   and this would likely cause issues later on
# - Final number of cases in dataset: 3081