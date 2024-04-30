from ultralytics import YOLO

model = YOLO('Models/best.pt')


results = model.predict('input_videos/08fd33_4.mp4',save=True)
print(results[0])
print('--------------------------------')
for box in results.boxes[0]:
    print(box)

