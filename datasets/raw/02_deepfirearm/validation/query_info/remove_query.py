import json

im_name = "M39+Enhanced+Marksman+Rifle_0044.jpg"

with open("ground_truth_info.json", "r") as f:
    gt_info = json.load(f)

gt_info.pop(im_name, None)
# print("{}".format(len(gt_info.keys())))

with open("ground_truth_info.json", "w") as f:
    json.dump(gt_info, f)