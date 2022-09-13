from flask import Flask, render_template, request, url_for
import os
import backend.Inference.ModelInference as inference





"""template_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
template_dir = os.path.join(template_dir, 'flask')
template_dir = os.path.join(template_dir, 'vista')
template_dir = os.path.join(template_dir, 'templates')


static_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
static_dir = os.path.join(static_dir, 'flask')
static_dir = os.path.join(static_dir, 'static')

static_dir = static_dir.replace("\\", "/")
print(static_dir)"""

app = Flask(__name__ ,template_folder='vista/templates/',  static_folder="static/")
#app.config['UPLOAD_FOLDER'] = static_dir
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Proyecto que para distribuir!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename

		img.save(img_path)

		p = inference.predict_label(img_path)
	return render_template("index.html", prediction = p, img_path =img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(host="0.0.0.0", debug=True)