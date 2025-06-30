import os
import yaml
import sys
import json
import os
import yaml
import sys
import json

def types_regions(filename, file_number):
    
    def load_yaml_file(filepath):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
        
    base_filename = os.path.basename(filename)
    base_path = "/home/norm/workspace/VIRAT/annotationsYML/"

    regions_file    = os.path.join(base_path, f"{base_filename}.regions.yml")
    types_file      = os.path.join(base_path, f"{base_filename}.types.yml")
    activities_file = os.path.join(base_path, f"{base_filename}.activities.yml")

    
    regions_data    = load_yaml_file(regions_file)
    types_data      = load_yaml_file(types_file)
    activities_data = load_yaml_file(activities_file)

    types_list = []
    for item in types_data:
        if "types" in item:
            entry = {
                "file": file_number, 
                "id": item["types"]["id1"], 
                "type": list(item["types"]["cset3"].keys())[0]
            }
            types_list.append(entry)

    regions_list = []
    for item in regions_data:
        if "regions" in item:
            entry = {
                "file": file_number,  
                "id": item["regions"]["id1"],
                "ts0": item["regions"]["ts0"],
                "poly0": item["regions"]["poly0"]
            }
            regions_list.append(entry)

    activities_list = []
    first_meta = activities_data[0]['meta']
    video_source = f"/home/norm/data/VIRAT/videos/{first_meta.split()[0]}.mp4"
    for item in activities_data:
        if "act" in item:
            actors = [actor["id1"] for actor in item["act"]["actors"]]
            entry = {
                "video_source": video_source, 
                "file": file_number,
                "activity id": item["act"]["id2"],
                "activity": list(item["act"]["act2"].keys())[0],
                "time range": item["act"]["timespan"][0]["tsr0"],
                "actors": actors
            }
            activities_list.append(entry)

    return types_list, regions_list, activities_list

def midpoint(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (sum(xs) / len(xs), sum(ys) / len(ys))

def bbox_from_poly(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    lefttop_x = min(xs)
    lefttop_y = min(ys)
    width = max(xs) - lefttop_x
    height = max(ys) - lefttop_y
    return lefttop_x, lefttop_y, width, height

def interesting_activities(ids_regions, ids_activities, ids_types): 
   
    regions_index = {}
    for region in ids_regions:
        ts0 = region['ts0']
        file = region['file']
        obj_id = region['id']
        if isinstance(ts0, list):
            start_time = ts0[0]
            end_time = ts0[-1]
            for t in range(start_time, end_time + 1):
                key = (obj_id, file, t)
                regions_index.setdefault(key, []).append(region['poly0'])
        else:
            key = (obj_id, file, ts0)
            regions_index.setdefault(key, []).append(region['poly0'])

    ids_types_lookup = {(entry["file"], entry["id"]): entry["type"] for entry in ids_types}
    all_activities = []

    for activity in ids_activities:
        """if activity["activity"] in activity_types_to_generate:
            file = activity["file"]
            min_time, max_time = activity["time range"][0], activity["time range"][1]
            video_source = activity["video_source"]
            one_activity = {
                "video_source": video_source,
                "file": file,
                "activity": activity["activity"],
                "start": min_time,
                "end": max_time,
                "activity_id": activity.get("activity id", None)
            }
            actors_info = []
            for actor in activity["actors"]:
                bbox_list = []
                for t in range(min_time, max_time + 1):
                    key = (actor, file, t)
                    if key in regions_index:
                        poly = regions_index[key][0]
                        pt = midpoint(poly)
                        lefttop_x, lefttop_y, bbox_width, bbox_height = bbox_from_poly(poly)
                        bbox_list.append({
                            "frame": t,
                            "mid_x": pt[0],
                            "mid_y": pt[1],
                            "bbox_lefttop_x": lefttop_x,
                            "bbox_lefttop_y": lefttop_y,
                            "bbox_width": bbox_width,
                            "bbox_height": bbox_height
                        })
                actor_type = ids_types_lookup.get((file, actor), "unknown")
                actor_info = {
                    "actor_id": actor,
                    "actor_type": actor_type,
                    "bbox_per_frame": bbox_list
                }
                actors_info.append(actor_info)
            one_activity["trajectory"] = actors_info
            all_activities.append(one_activity)"""
        
        file = activity["file"]
        min_time, max_time = activity["time range"][0], activity["time range"][1]
        video_source = activity["video_source"]
        one_activity = {
            "video_source": video_source,
            "file": file,
            "activity": activity["activity"],
            "start": min_time,
            "end": max_time,
            "activity_id": activity.get("activity id", None)
        }
        actors_info = []
        for actor in activity["actors"]:
            bbox_list = []
            for t in range(min_time, max_time + 1):
                key = (actor, file, t)
                if key in regions_index:
                    poly = regions_index[key][0]
                    pt = midpoint(poly)
                    lefttop_x, lefttop_y, bbox_width, bbox_height = bbox_from_poly(poly)
                    bbox_list.append({
                        "frame": t,
                        "mid_x": pt[0],
                        "mid_y": pt[1],
                        "bbox_lefttop_x": lefttop_x,
                        "bbox_lefttop_y": lefttop_y,
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height
                    })
            actor_type = ids_types_lookup.get((file, actor), "unknown")
            actor_info = {
                "actor_id": actor,
                "actor_type": actor_type,
                "bbox_per_frame": bbox_list
            }
            actors_info.append(actor_info)
        one_activity["trajectory"] = actors_info
        all_activities.append(one_activity)

    return all_activities

def store_trajectories(activities, output_json):
    trajectories = {}
    for idx, act in enumerate(activities):
        event_id = act.get("activity_id", idx)
        key = f"{act['file']}_{event_id}"
        trajectories[key] = {
            "event_id": event_id,
            "trajectory": act["trajectory"],
            "start_frame": act["start"],
            "end_frame": act["end"],
            "activity": act["activity"],
            "activity_id": event_id,
            "source": act["video_source"]
        }
    with open(output_json, "w") as f:
        json.dump(trajectories, f, indent=4)
    print(f"Trajectories saved to {output_json}")

def main(filenames_path):
    with open(filenames_path, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    ids_types = []
    ids_regions = []
    ids_activities = []

    for file_number, base_filename in enumerate(filenames):
        print(f"Processing file {file_number+1}/{len(filenames)}: {base_filename}")
        try:
            types_list, regions_list, activities_list = types_regions(base_filename, file_number)
            ids_types.extend(types_list)
            ids_regions.extend(regions_list)
            ids_activities.extend(activities_list)
        except Exception as e:
            print(f"Error processing {base_filename}: {e}")

    print("All files processed.")
    all_activities = interesting_activities(ids_regions, ids_activities, ids_types)
    output_json = "trajectories_Extended.json"
    store_trajectories(all_activities, output_json)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py filenames_path.txt")
    else:
        main(sys.argv[1])



