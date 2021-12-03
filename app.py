


import os

from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image

from PIL import Image
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

import torchvision as tv
import shutil




ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

class MedNet(nn.Module):
    def __init__(self,xDim,yDim,numC): 
        super(MedNet,self).__init__()  
        
        numConvs1 = 5
        convSize1 = 7
        numConvs2 = 10
        convSize2 = 7
        numNodesToFC = numConvs2*(xDim-(convSize1-1)-(convSize2-1))*(yDim-(convSize1-1)-(convSize2-1))

        self.cnv1 = nn.Conv2d(1, numConvs1, convSize1)
        self.cnv2 = nn.Conv2d(numConvs1, numConvs2, convSize2)
        
        fcSize1 = 400
        fcSize2 = 80
        
        self.ful1 = nn.Linear(numNodesToFC,fcSize1)
        self.ful2 = nn.Linear(fcSize1, fcSize2)
        self.ful3 = nn.Linear(fcSize2,numC)
        
    def forward(self,x):
       
        
        x = F.elu(self.cnv1(x)) # Feed through first convolutional layer, then apply activation
        x = F.elu(self.cnv2(x)) # Feed through second convolutional layer, apply activation
        x = x.view(-1,self.num_flat_features(x)) # Flatten convolutional layer into fully connected layer
        x = F.elu(self.ful1(x)) # Feed through first fully connected layer, apply activation
        x = F.elu(self.ful2(x)) # Feed through second FC layer, apply output
        x = self.ful3(x)        # Final FC layer to output. No activation, because it's used to calculate loss
        return x

    def num_flat_features(self, x):  # Count the individual nodes in a layer
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


app = Flask(__name__)
app.config['SECRET_KEY']  = "secret key"
UPLOAD_FOLDER = os.path.join('static', 'uploadsImage')
app.config['UPLOAD_FOLDER']  = UPLOAD_FOLDER
class_names =['AbdomenCT', 'BreastMRI', 'ChestCT', 'CXR', 'Hand', 'HeadCT']

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prediction(image):

	
	toTensor = tv.transforms.ToTensor()
	def scaleImage(x):          # Pass a PIL image, return a tensor
		y = toTensor(x)
		if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
			yu_çày = (y - y.min())/(y.max() - y.min()) 
		z = y - y.mean()        # Subtract the mean value of the image
		return z
	img = scaleImage(Image.open(image))
	img =torch.unsqueeze(img, 0)
	model = torch.load( "saved_model")
	with torch.no_grad():
		num_class=int(np.argmax(model(img)))
		return class_names[num_class]
	
@app.route('/')
def upload_form():
	return render_template('uploadImage.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		pred= "Pas de prédiction"
		#pred =prediction(file)
		
		#flash(os.listdir(app.config['UPLOAD_FOLDER']))


	for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
		for f in files:
			os.unlink(os.path.join(root, f))
		for d in dirs:
			shutil.rmtree(os.path.join(root, d))
	

		return render_template('uploadImage.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

#@app.route('/static/uploadsImage/<filename>')
#def display_image(filename):
#	#print('display_image filename: ' + filename)
#	#full_filename = os.path.join(app.config['UPLOAD_FOLDER'],'mapAsiawhite.jpg')
#	return redirect(url_for('static', filename='/uploadsImage/' + filename), code=301)

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000)