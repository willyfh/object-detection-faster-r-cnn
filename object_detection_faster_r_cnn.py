import cv2
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# download pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# class names from PyTorch's official Docs
CATEGORY_NAMES = [
	'__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
	'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
	'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
	'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
	'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
	'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
	'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
	'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
	'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
	'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
	'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
	'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# color for each category
CATEGORY_COLORS = [
	(0,0,0), (0,255,0), (0,255,0), (255,0,0), (0,255,0), (0,255,0), (0,0,255),
	(0,255,0), (0,255,0), (0,255,0), (0,0,255), (0,255,0), (0,255,0), (0,255,0),
	(0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
	(0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
	(0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
	(0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
	(0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
	(0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
	(0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
	(0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
	(0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0),
	(255,0,255), (0,255,0), (0,255,0), (0,255,0), (0,255,0), (0,255,0)
]

# prediction
def get_prediction(img_path, threshold):
	img = Image.open(img_path) # Load the image
	transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
	img = transform(img) # Apply the transform to the image
	pred = model([img]) # Pass the image to the model
	pred_class = [CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())] # Get the Prediction Score
	pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())] # Bounding boxes
	pred_score = list(pred[0]['scores'].detach().numpy())
	pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1] # Get list of index with score greater than threshold.
	pred_boxes = pred_boxes[:pred_t+1]
	pred_class = pred_class[:pred_t+1]
	return pred_boxes, pred_class

# draw result
def object_detection_api(img_path, threshold=0.5, rect_th=3, text_size=1, text_th=2):
	boxes, pred_cls = get_prediction(img_path, threshold) # Get predictions
	img = cv2.imread(img_path) # Read image with cv2
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to RGB
	for i in range(len(boxes)):
		cv2.rectangle(img, boxes[i][0], boxes[i][1],color=CATEGORY_COLORS[CATEGORY_NAMES.index(pred_cls[i])], thickness=rect_th) # Draw Rectangle with the coordinates
		cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,0,0),thickness=3) # Write the prediction class
		cv2.putText(img,pred_cls[i], boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,255,255),thickness=text_th) # Write the prediction class
	plt.figure(figsize=(20,30)) # display the output image
	plt.imshow(img)
	plt.xticks([])
	plt.yticks([])
	plt.show()

object_detection_api('./sample_image.jpg', threshold=0.8)