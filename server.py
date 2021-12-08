from flask import Flask, send_file, redirect, request
from comicgen import *

app = Flask(__name__, static_url_path='', static_folder='static')
CG = ComicGenerator('models/final/14085_G.pth')
prefix = 'models/final/'
suffix = '_G.pth'

@app.route('/imgen')
def image():
    CG.generateImage()
    return send_file('fakeim.png')

@app.route('/change', methods=['GET'])
def changeGen():
    arg = request.args['newGen']
    if arg != 'selected':
        CG.update(prefix+arg+suffix)
    return "<h1></h1>"

@app.route('/')
def home():
    return redirect('/index.html')


if __name__ == '__main__':
    app.run(debug=True)
