import sqlite3
import FileParser
import datetime
import requests
from warnings import warn
from Worker import Worker

def load_data_from_database(DB_path):
    pass

class Database:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()

    def create_table(self):
        try:
            query = """
            CREATE TABLE stem(
                Awal varchar(100),
                Akhir varchar(100)
            );
            """
            self.c.execute(query)
            query = """
            CREATE TABLE berita (
                Source varchar(500),
                Title varchar(200),
                Url varchar(1000),
                Text varchar(50000),
                Cleaned varchar(50000),
                Date int,
                Clock int,
                Sentiment int
            );
            """
            self.c.execute(query)

            query = """
            CREATE TABLE harga(
                Harga float,
                Date int,
                Pembukaan boolean
            );
            """
            self.c.execute(query)
            return "Database dibuat"
        except:
            return "Database sudah dibuat"

    def update_price(self):
        #Tambah data entry harga saham
        n = 0
        file = open('harga.csv')
        data = file.readlines()
        for I in data[1:]:
            line = I.split(',')
            date = int(line[0].replace('-',''))
            try:
                opening = float(line[1])
                closing = float(line[-3])
            except:
                # Artinya kosong. Tidak ada perdagangan. Entri selanjutnya
                continue
            self.c.execute("SELECT * FROM harga WHERE Date= ?",(date,))
            d = self.c.fetchall()
            # Jika len(d)==0, artinya entri baru. Masukan
            if len(d)==0 :
                self.c.execute("INSERT INTO harga VALUES (?, ?, ?);",(opening,date,1))
                self.c.execute("INSERT INTO harga VALUES (?, ?, ?);",(closing,date,0))
                n += 1
        self.conn.commit()
        return n

    def update_news(self, beginDate, endDate):
        """
        Tambah data entri berita

        Input:
        beginDate -- integer
        Awal tanggal pencarian berita dengan format
        beginDate = beginyear * 10000 + beginMonth * 100 + beginDay
        
        endDate -- integer
        Akhir tanggal pencarian berita dengan format
        beginDate = endyear * 10000 + endMonth * 100 + endDay
        
        Output:
        Sebuah integer menunjukan berapa banyak data berita yang ditambah
        """
        if beginDate > endDate:
            raise ValueError("beginDate (%d) harus secara kronologis sebelum endDate (%d)"
                %(beginDate,endDate))

        beginDate = datetime.datetime(beginDate//10000,beginDate//100%100,beginDate%100)
        endDate = datetime.datetime(endDate//10000,endDate//100%100,endDate%100)

        for I in FileParser.FileParser.parserList:
            parser = FileParser.FileParser(I)
            print(I)
            date = beginDate
            worker = Worker()
            while date <= endDate:
                newsList = parser.extractPublikasi(date.year*10000+date.month*100+date.day)
                print(date.year*10000+date.month*100+date.day)
                for J in newsList:
                    #Cek apakah artikel sudah pernah di-crawl?
                    self.c.execute("SELECT * FROM berita where Url = ?",(J['url'],))
                    d = self.c.fetchall()
                    #Kalau belum, masukan
                    if len(d) == 0:
                        worker.addOrder(J['url'])
                result = worker.getData()
                print(len(result))
                for J in newsList:
                    if J['url'] in result:
                        r = result[J['url']]
                        nextTwoSession = self.find_session(J['tanggal'], J['jam'])
                        # Sentimen positif = 1
                        # Sentimen netral = 0
                        # Sentimen negatif = -1

                        # Untuk awal, diasumsikan semua berita ketika harga naik adalah sentimen
                        # positif dan diasumsikan semua berita ketika harga turun adalah
                        # sentimen negatif
                        sentiment = int(self.compare_index(nextTwoSession) > 0) * 2 + -1
                        d = parser.extractData(r)
                        if d != None:
                            self.c.execute("INSERT INTO berita VALUES (?,?,?,?,?,?,?,?)",
                                (J['sumber'],J['judul'],J['url'],d,'',J['tanggal'],J['jam'],sentiment))
                date += datetime.timedelta(1)
                worker.reset()
                self.conn.commit()
                
    def compare_index(self, date):
        '''
        Mengcompare antara harga gabungan indeks saham pada milestone 1
        dan harga gabungan indek saham pada milestone 2

        INPUT:
        Sebuah list hasil keluaran cariSesi()

        OUTPUT:
        Sebuah bilangan yang menunjukan selisih antara milestone 1 dan milestone 2
        dengan output = harga indeks saham pada milestone 2 - milestone 1
        '''
        # Sanitasi data
        if date[0] > date[1]:
            raise ValueError("Input milestone 1 harus kronologis sebelum milestone 2")
        mileStoneValue = []
        for I in range(2):
            self.c.execute("SELECT Harga from harga where Date = ?",(date[I],))
            d = self.c.fetchall()
            if len(d) != 1:
                raise ValueError("Tidak bisa mencari data valid untuk sesi %s" %
                                (date[I]))
            mileStoneValue.append(d[0][0])
        return mileStoneValue[1] - mileStoneValue[0]

    def find_session(self, date, clock):
        d = datetime.datetime(date//10000,date//100%100,date%100)
        if clock > 945: 
            d += datetime.timedelta(1)
        answer = []
        while len(answer) < 2: 
            n = d.isoweekday()
            if n > 5: 
                d += datetime.timedelta(8-n)
            self.c.execute("SELECT * FROM harga where Date = ?",(d.year*10000+d.month*100+d.day,))
            #print(d.year*10000+d.month*100+d.day)
            r = self.c.fetchall()
            if len(r) > 0:
                answer.append(d.year*10000+d.month*100+d.day)
            d += datetime.timedelta(1)
        return answer