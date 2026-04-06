from rgbt import RGBT210

rgbt210 = RGBT210()

# Register your tracker
rgbt210(
    tracker_name="HAFTrack",
    result_path="", 
    bbox_type="ltwh",
    prefix="")

# Evaluate multiple trackers

sr_dict = rgbt210.SR()
print(sr_dict["HAFTrack"][0])

pr_dict = rgbt210.PR()
print(pr_dict["HAFTrack"][0])