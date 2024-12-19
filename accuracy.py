import json
import multiprocessing


results = json.load(open("recognized.json"))
mapping = json.load(open("label_to_id.json"))

def get_accuracy(results):
    correct = 0
    for result in results:
        if result["actual_label"] == result["mapped_label"]:
            correct += 1
    return correct/len(results)

def get_accuracy_for_class(results, class_name):
    correct = 0
    total = 0
    for result in results:
        if result["actual_label"] == class_name:
            total += 1
            if result["actual_label"] == result["mapped_label"]:
                correct += 1
    return correct/total

def get_consistency(results):
    scores = {}
    for result in results:
        if result["actual_label"] not in scores:
            scores[result["actual_label"]] = {}
        if result["mapped_label"] not in scores[result["actual_label"]]:
            scores[result["actual_label"]][result["mapped_label"]] = 0
        scores[result["actual_label"]][result["mapped_label"]] += 1
    return scores

def get_consistency_score(consistency):
    total = 0
    correct = 0
    for key in consistency:
        for key2 in consistency[key]:
            total += consistency[key][key2]
        correct += max(consistency[key].values())
    return correct/total

def get_folder_lengths():
    path = "train_images/train/"
    import os
    folders = os.listdir(path)
    folder_lengths = {}
    for folder in folders:
        folder_lengths[folder] = len(os.listdir(path+folder))
    return folder_lengths

def get_result_lengths(results):
    result_lengths = {}
    for result in results:
        if result["recognized"] not in result_lengths:
            result_lengths[result["recognized"]] = 0
        result_lengths[result["recognized"]] += 1
    result_lengths = dict(sorted(result_lengths.items(), key=lambda item: item[1], reverse=True))
    return result_lengths

def match_lengths(result_lengths, folder_lengths):
    for key in folder_lengths:
        folder_len = folder_lengths[key]
        if folder_len not in result_lengths.values():
            print("No match for ", key, " with length ", folder_len)
        else:
            result_key = [k for k, v in result_lengths.items() if v == folder_len][0]
            print("Match for ", key, " with length ", folder_len, " in ", result_key)

print(match_lengths(get_result_lengths(results), get_folder_lengths()))
with open("lengths.json", "w") as f:
    json.dump(get_result_lengths(results), f)
with open("lengths2.json", "w") as f:
    json.dump(get_folder_lengths(), f)
#print("Overall accuracy: ", get_accuracy(results))

#for class_name in mapping:
    #print(class_name, " ", get_accuracy_for_class(results, class_name))
    
#consistency = get_consistency(results)
#print("Consistency score: ", get_consistency_score(consistency))
#print(consistency)



