from flask import Flask, send_file
app = Flask(__name__)

@app.route('/')
def image():
    return send_file('image/attn_gf1.png')

if __name__ == '__main__':
    app.run(debug=True)
