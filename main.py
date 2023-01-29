from flask import Flask , render_template ,request
import os
from matplotlib import pyplot as plt
from style_transfer import load_image, model, stylize, im_convert


app = Flask(__name__)
UPLOAD_FOLDER = './static/image/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
style =""
@app.route("/")
def home():
	return render_template("index.html")


@app.route("/processed", methods=['POST'])

def upload_file():
			content = request.files['file']
			style = request.form.get('style')
			content.save(os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg'))
			#load in content and style image
			content = load_image('./static/image/upload/content.jpg')
		 	#Resize style to match content, makes code easier
			style = load_image('./static/image/s'+ style+'.jpg', shape=content.shape[-2:])
			vgg = model()
			target = stylize(content,style,vgg)
			x = im_convert(target)
			plt.imsave(app.config['UPLOAD_FOLDER']+'/target.jpg', x)

			return render_template('processed.html')

							

if __name__ =="__main__":
	app.run(debug=True)
