from flask import Flask, render_template, request, url_for
import backend.Inference.ModelInference as inference



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

		p, grad_cam, proba = inference.predict_label(img_path)
	return render_template("index.html", prediction = p, img_path =img_path, heatmap_path = grad_cam, probabilidad_diagnostico = proba)


if __name__ =='__main__':
	#app.debug = True
	app.run(host="0.0.0.0", debug=True)