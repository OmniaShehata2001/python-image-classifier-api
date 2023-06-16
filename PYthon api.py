from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms as T
import torchvision

# Replace this with your own classification function


def classify_image(image):
    # Load and preprocess the image
    
    
    img_transforms = {
    'test': T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),}
    img = image
    
    if img.mode != 'RGB':
        img = img.convert('RGB')

    input_tensor = img_transforms['test'](img)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Load the PyTorch model
    model=torchvision.models.mobilenet_v3_large(pretrained=True)
    state_dict = torch.load("model.pt",map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()
    
   

    # Pass the image through the model and get the output
    with torch.no_grad():
        output = model(input_batch)  # Pass the preprocessed image tensor to the model
    _, preds = torch.max(output, 1)
    return preds.item()

   

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify():
    # Receive base64 encoded image from the request
    image_data = request.json.get('image')
    image_bytes = base64.b64decode(image_data)

    # Convert the image bytes to a PIL Image object
    image = Image.open(io.BytesIO(image_bytes))

    # Call your classification function
   
    brainmiled={0:"MildDemented",1:"ModerateDemented",2:"NonDemented",3:"VeryMildDemented"}
    result = classify_image(image)
    result=brainmiled[result]
    print(result)
    # Return the classification result as JSON
    return jsonify({'Degree of demented': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
