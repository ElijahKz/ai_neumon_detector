# DesignPattern


Hola! Bienvenido a la herramienta para la detección rápida de neumonía
Deep Learning aplicado en el procesamiento de imágenes radiográficas de tórax en formato jpeg con el fin de clasificarlas en 3 categorías diferentes:
<hr/>
Neumonía Bacteriana

Neumonía Viral

Sin Neumonía

Aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.
Diseño
<hr/>
<img src="https://ibb.co/dLz8nhy" alt="design" width="50%" height="50%">

¿Cómo correr este repositorio?
<hr/>
Instala docker en tu máquina. Luego instala docker compose de la siguiente forma:.<br/>
*sudo apt-get update<br/>
*sudo apt-get install docker-compose-plugin<br/>
Clona la carpeta de este repositorio y luego ejecuta el siguiente commando.<br/>
<strong>docker compose up</strong>. Y listo! <br/>
Obtendrás un mensaje como el siguiente.
<p>
flask-web-1    |  * Debug mode: off<br/>
flask-web-1    |  * Running on all addresses (0.0.0.0)<br/>
flask-web-1    |  * Running on http://127.0.0.1:5000<br/>
flask-web-1    |  * Running on http://172.21.0.3:5000<br/>
</p>

Abre la dirección http://172.21.0.3:5000 en tu navegador y a predecir!
<hr/>
Material con fines educativos sin ánimo de lucro integrantes.<br/>
* Sebastian Amilkar<br/>
* Milmax 