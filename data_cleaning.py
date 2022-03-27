import requests
import pandas as pd
import csv

# Calculates the average of a list
def avg(lst):
    lst_len = len(lst)
    if lst_len != 0:
        return sum(lst) / lst_len
    else:
        return lst_len

# Gets the list of justice ideologies at row n of data
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
        print(f"Error retrieving ideologies, skipping line {n}.")
        skip_nums.append(n)
        return []

# Gets a list containing the lists of justice ideologies
def get_ideologies_list():
    output = []
    for i in range(3303):
        output.append(get_ideologies(i))
    return output

# Change issue areas with no data type to "unknown"
def clean_issue_area(issue):
    if type(issue) != str:
        return "Unknown"
    else:
        return issue

# Cleans the given fact (still keeps <a> elements as there isn't an easy way to remove those)
# Keeps â€™ in the text rather than replacing it with '
def clean_fact(fact):
    fact = fact.replace("<p>", "").replace("</p>", "")[:-1]
    fact = fact.replace("<em>", "").replace("</em>", "").replace("<i>", "").replace("</i>", "")
    return fact.replace('<p dir="ltr">', "").replace('<p class="p1">', "")

# Adds rows to the skip list that have no value for which side wins, and also returns the list of boolean data
def clean_first_party_wins():
    bool_list = data["first_party_winner"].to_list()
    for i in range(len(bool_list)):
        if type(bool_list[i]) != bool:
            skip_nums.append(i)
            print(f"Skipping row {i} as we don't have data on which side won.")
    return bool_list
                       
# Creates a cleaned csv file with all the data we want
def main():

    # Creates lists for all the data and adds indecies to the skip_nums list that need to be skipped
    ideologies = get_ideologies_list()
    avg_ideologies = [avg(x) for x in ideologies]
    issue_areas = [clean_issue_area(x) for x in data["issue_area"].to_list()]
    facts = [clean_fact(x) for x in data["facts"].to_list()]
    first_wins = clean_first_party_wins()
    first_parties = data["first_party"].to_list()
    second_parties = data["second_party"].to_list()
    case_names = data["name"].to_list()
    docket_nums = data["docket"].to_list()

    # Skipping lines where the ideology data is innacurate
    for i in range(len(avg_ideologies)):
        if (not i in skip_nums) and (avg_ideologies[i] == 0):
            skip_nums.append(i)
            print(f"Skipping line {i} due to not being able to get accurate ideologies")

    # Writes the data to a new csv file called "clean_data.csv"
    with open("clean_data.csv", "w", newline="") as csvfile:
        fields = ['case_name', 'docket_num', 'first_party', 'second_party', 'facts', 
        'first_party_won', 'issue_area', 'ideologies', 'avg_ideology']
        writer = csv.DictWriter(csvfile, fieldnames = fields)
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
                    skip_nums.append(i)

    print(f"Finished the data cleaning. A total of {len(skip_nums)} lines were skipped, making the total number of cases: {3303 - len(skip_nums)}.")


if __name__ == "__main__":
    # Reads the data
    data = pd.read_csv("justice.csv")

    # Creates a list for rows to be skipped in the new clean data
    skip_nums = []

    # Runs the main program
    main()
