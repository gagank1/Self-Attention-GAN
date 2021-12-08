from flask import Flask, send_file
import io

app = Flask(__name__)

@app.route('/')
def image():
    send_file('image/attn_gf1.png')