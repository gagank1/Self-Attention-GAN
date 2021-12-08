from flask import Flask, send_file, redirect, request
from PIL import Image
from comicgen import *
import io

app = Flask(__name__, static_url_path='', static_folder='static')

prefix = 'models/final/'
suffix = '_G.pth'
CGMap = {
    '14085': 0,
    '17215': 1,
    '21910': 2,
    '29735': 3
}
CGs = [ComicGenerator(prefix+num+suffix) for num in CGMap]


@app.route('/imgen', methods=['GET'])
def image():
    arg = request.args['genNum']
    if arg in CGMap:
        img = Image.fromarray(CGs[CGMap[arg]].generateImage(), mode='RGB')
        file_object = io.BytesIO()
        img.save(file_object, 'PNG')
        file_object.seek(0)
        return send_file(file_object, mimetype='image/PNG')
    else:
        return redirect('/dummy.png')

@app.route('/')
def home():
    return redirect('/index.html')


if __name__ == '__main__':
    app.run(debug=True)
