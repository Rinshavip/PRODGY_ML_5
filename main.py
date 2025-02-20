from ultralytics import YOLO
import pandas as pnd

try:
    food_info = pnd.read_csv("Food_Calorie.csv",index_col=False)
except FileNotFoundError:
    print("File Not found")
    exit()

#Loding Yolo model
model = YOLO('yolov8n.pt')

results = model(source=0,show=True,conf=0.5,save =True,verbose=False,stream=True)

food_info['Name'] = food_info['Name'].str.lower()

for result in results:
    bboxes = result.boxes
    if bboxes:
        for bbox in bboxes:
            cls_id = int(bbox.cls)
            class_name = model.names[cls_id].lower()

            if class_name in food_info['Name'].values:
                calorie = food_info[food_info['Name']==class_name].loc[0]
                print(f"found {calorie['Name']},{calorie['Amount']} {calorie['Name']} has {calorie['Calorie']} Calories")

