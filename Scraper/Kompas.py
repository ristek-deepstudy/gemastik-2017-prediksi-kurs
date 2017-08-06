import requests
from bs4 import BeautifulSoup
from warnings import warn

class Kompas:
    '''
    Scraper untuk kompas.com
    '''
    def extractData(htmlFile):
        """
        Mengekstrak isi berita kompas.com
        INPUT:

        htmlFile -- request.models.Response
        Sebuah request.models.Response yang diekstrak dari laman
        berita kompas.com

        OUTPUT:

        me-return sebuah string berisi berita yanghtmlFile kompas.com
        yang sudah diekstrak
        """
        try:
            soup = BeautifulSoup(htmlFile.content, 'html.parser')
            text = soup.find('div','read__content').getText()
            text += " ".join([I.getText() for I in soup.find_all('p')])
            return  text.replace('KOMPAS.com - ','')
        except:
            return None
    def extractPublikasi(date):
        """
        Meng-ekstrak semua artikel bisnis keuangan yang
        di hasilkan pada tanggal itu di Kompas.com

        INPUT:

        data -- integer
        Sebuah angka dengan format
        data = year * 10000 + month * 100 + day

        OUTPUT:

        Sebuah list yang isinya dictionary. Setiap dictionary isinya

        sumber : "Kompas"
        tanggal : integer dari variable date
        judul : judul dari artikel
        url : pranala artiker
        jam : jam publikasi dalam bentuk integer dalam format

        jam = 100*h + m
        """
        year = date // 10000
        month = date // 100 % 100
        day = date % 100
        url = "http://bisniskeuangan.kompas.com/search/%d-%d-%d/%d"
        result = []
        ids = 1
        while True:
            # Fungsi ids adalah untuk mengakses halaman ke-ids di indeks
            for J in range(3):
                index = requests.get(url%(year,month,day,ids))
                if index.status_code == 200:
                    break
            else:
                warn("Kompas gagal mengesktrak indeks pada %d/%d/%d pada halaman %d",(year,month,day,ids+1))
                break
            indexSoup = BeautifulSoup(index.content, 'html.parser')
            # kompas.com termasuk yang paling gampang dicari datanya
            articleList = indexSoup.find_all('div','article__title article__title--medium')
            dateList = indexSoup.find_all('div','article__date')
            # Kalau tidak ketemu berhati habis.
            if len(articleList) == 0:
                break
            # Pindak ke halaman selanjutnya
            dateList = [int(I.getText().split(" ")[1].replace(":","")) for I in dateList]
            urlList = [I.find('a')['href'] for I in articleList]
            titleList = [I.find('a').getText() for I in articleList]
            for J in range(len(urlList)):
                result.append({'judul':titleList[J],
                   'url':urlList[J],
                   'tanggal':date,
                   'sumber':"Kompas",
                   'jam':dateList[J]})
            ids += 1
        return result 