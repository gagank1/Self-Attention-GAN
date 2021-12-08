from flask import Flask, send_file, redirect, request
from comicgen import *
import threading

app = Flask(__name__, static_url_path='', static_folder='static')

prefix = 'models/final/'
suffix = '_G.pth'
CGMap = {
    '14085': 0,
    '17215': 1,
    '21910': 2,
    '29735': 3
}
CGs = [ComicGenerator(prefix+num+suffix, CGMap[num]) for num in CGMap]
sem = threading.Semaphore()

@app.route('/imgen', methods=['GET'])
def image():
    arg = request.args['genNum']
    if arg in CGMap:
        sem.acquire()
        CGs[CGMap[arg]].generateImage()
        resp = send_file('fakeim'+str(CGMap[arg])+'.png')
        sem.release()
    else:
        resp = redirect('/dummy.png')
    return resp

# @app.route('/change', methods=['GET'])
# def changeGen():
#     arg = request.args['newGen']
#     if arg != 'selected':
#         CG.update(prefix+arg+suffix)
#     return "<h1></h1>"

@app.route('/')
def home():
    return redirect('/index.html')


if __name__ == '__main__':
    app.run(debug=True)
