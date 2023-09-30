from ultralytics import YOLO

model = YOLO('models/model-seg.pt')  # load a pretrained model (recommended for training)

results = model('data/captiva.jpg')

print(results[0].masks)
# results.save()  # or results.save('output.jpg')
# results.crop()  # crop the image to the bounding box
# results.pandas().xyxy[0]  # return bounding box coordinates in a Pandas DataFrame