from flask import Flask, send_file, redirect
from comicgen import *

app = Flask(__name__, static_url_path='', static_folder='static')
CG = ComicGenerator('models/final/15650_G.pth')

@app.route('/imgen')
def image():
    CG.generateImage()
    return send_file('fakeim.png')

@app.route('/')
def home():
    return redirect('/index.html')

if __name__ == '__main__':
    app.run(debug=True)
