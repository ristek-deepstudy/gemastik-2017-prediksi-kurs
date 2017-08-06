import requests
from bs4 import BeautifulSoup
from warnings import warn

class Detik:
    '''
    Scraper untuk detik.com
    '''
    def extractData(htmlFile):
        """
        Mengekstrak isi berita detik.com
        INPUT:

        htmlFile -- request.models.Response
        Sebuah request.models.Response yang diekstrak dari laman
        berita kompas.com

        OUTPUT:

        me-return sebuah string berisi berita yanghtmlFile detik.com
        yang sudah diekstrak
        """
        try:
            soup = BeautifulSoup(htmlFile.content, 'html.parser')
            text =  soup.find('div','detail_text').getText()
            text = text.split('\n\n\n\n')[0]
            return text
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

        sumber : "Detik"
        tanggal : integer dari variable date
        judul : judul dari artikel
        url : pranala artiker
        jam : jam publikasi dalam bentuk integer dalam format

        jam = 100*h + m
        """
        year = date // 10000
        month = date // 100 % 100
        day = date % 100    
        url = "https://finance.detik.com/indeks/?date="+str(day)+"%2F"+str(month)+"%2F"+str(year)
        result = []
        for J in range(3):
            r = requests.get(url)
            if r.status_code == 200:
                    break
        else:
            warn("Kompas gagal mengesktrak indeks pada %d/%d/%d pada halaman %d",(year,month,day,ids+1))
            return None
        soup = BeautifulSoup(r.content,'html.parser')
        articleList = soup.find_all('div','desc_idx ml10')
        for I in articleList:
            title = I.find('h2').getText()
            url = I.find('a')['href']
            clock = I.find("span").getText().split(",")[1].split(" ")[1].split(":")
            clock = [int(I) for I in clock]
            clock = clock[0] * 100 + clock[1]
            result.append({"sumber":"Detik","tanggal":date,"judul":title,"url":url,"jam":clock})
        return result