from sklearn.feature_extraction.text import CountVectorizer
import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import sqlite3
class Stemizer:
    '''
    Fungsi dari stemizer adalah mengurus segala tetek bengek pre-processing kata
    seperti pengubahan dari bentuk berimbuhan menjadi kata dasar.
    '''
    def __init__(self):
        # Hasil transformasi kata disimpan di tabel stem untuk menghemat komputasi
        self.conn = sqlite3.connect('berita.db')
        self.c = self.conn.cursor()
    def bangunStem(self):
        '''
        Membangun database kata yang distem
        Note : Jika ini pertama kali memanggil bangunStem() maka proses komputasi
        ini akan sangat lama (>= 10 jam)
        '''
        self.c.execute("SELECT Text FROM berita;")
        result = self.c.fetchall()
        #TODO: Cari cara yang lebih elegan untuk meng-build daftar kata daripada dengan CountVectorizer
        counter = CountVectorizer()
        counter.fit([I[0] for I in result])
        result = list(counter.vocabulary_.keys())
        # Ada beberapa token yang mengandung angka. Token tersebut tentu tidak bisa di-stem jadi 
        # kita buang saja
        result = [I for I in result if I.isalpha()]
        # Kalau ini bukan pertama kali kita menjalankan bangunStem(), tentu sudah ada yang sudah di-stem
        # Untuk menghemat komputasi, jangan di-stem lagi
        self.c.execute("SELECT * FROM stem;")
        hasil = self.c.fetchall()
        daftarKata = []

        for I in hasil:
            daftarKata.append(I[0])
            daftarKata.append(I[1])
        
        belumDiStem = list(set(result) - set(daftarKata))

        t = tqdm.tqdm(total = len(belumDiStem))

        # create stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        for I in belumDiStem:
            hasil = stemmer.stem(I)
            if I != hasil:
                self.c.execute("INSERT INTO stem VALUES (?,?)",(I,hasil))
                # Commit itu memang boros, tetapi gunanya agar bisa di pause operasinya
                self.conn.commit()
            t.update()
    def bersihkanTeks(self,mode="soft"):
        """
        Mengkonversi kalimat berita menjadi kalimat yang sudah di-stem
        sesuai database stem yang dimiliki

        Input:

        mode -- String
        Menentukan berita mana yang akan dibersihkan.
        Jika "soft", berita yang belum dibersihkan akan dibersihan.
        Jika "hard", berita yang sudah dibersihkan akan dibersihkan.

        default ke "soft"
        """
        if mode == "soft":
            self.c.execute("SELECT Url,Text FROM berita WHERE Cleaned = ''")
        elif mode == "hard":
            self.c.execute("SELECT Url,Text FROM berita")
        else:
            raise ValueError('mode harusnya "soft" atau "hard" tetapi nilainya %s' % (mode))
        hasil = self.c.fetchall()[24725:-10592]

        self.c.execute("SELECT * FROM Stem")
        result = self.c.fetchall()

        pemetaan = {result[J][0]:result[J][1] for J in range(len(result))}
        t = tqdm.tqdm(total=len(hasil))
        for I in hasil:
            url = I[0]
            teks = I[1]
            #Ubah segala hal yang bukan alfabet menjadi spasi

            teksYangDiubah = ""
            for I in teks:
                if I.isalpha():
                    teksYangDiubah += I.lower()
                else:
                    teksYangDiubah += " "
            for J in pemetaan:
                if " "+J+" " in teksYangDiubah:
                    teksYangDiubah = teksYangDiubah.replace(" "+J+" "," "+pemetaan[J]+" ")
            self.c.execute("UPDATE berita SET Cleaned='%s' WHERE Url='%s'" % (teksYangDiubah,url))
            self.conn.commit()
            t.update()