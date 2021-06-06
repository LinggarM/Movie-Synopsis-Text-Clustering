from flask import Flask, request, render_template
import kmeans

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
	if request.method == 'POST':
		# Retrieve Data
		jumlah_tugas = request.form['jumlah_tugas']
		nama_tugas = request.form['nama_tugas']
		skala_prioritas = request.form['skala_prioritas']
		waktu_pengerjaan = request.form['waktu_pengerjaan']
		waktu = request.form['waktu']
		
		# Render
		return render_template('index.html', data = knapsack.kalkulasi(jumlah_tugas, nama_tugas, skala_prioritas, waktu_pengerjaan, waktu))
	return render_template("index.html", data="")

if __name__ == "__main__":
    app.run()