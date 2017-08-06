import requests
from bs4 import BeautifulSoup
from warnings import warn

class Liputan6:
    '''
    Scraper untuk liputan6.com
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
            text =  soup.find('div','article-content-body__item-content').getText()
            text = text.replace(soup.find('div','baca-juga').getText(),"")
            return text.replace('Liputan6.com, ','')
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
        ids = 1
        result = []
        if month >= 10:
            month = str(month)
        else:
            month = "0" + str(month)
        if day >= 10:
            day = str(day)
        else:
            day = "0" + str(day)
        url = "http://bisnis.liputan6.com/indeks/%d/%s/%s?page=%d"
        while True:            
            for J in range(3):
                r = requests.get(url%(year,month,day,ids))
                print(url%(year,month,day,ids))
                if r.status_code == 200:
                    break
                else:
                    print(r.status_code)
            else:
                warn("Liputan6 gagal mengesktrak indeks pada %d/%d/%d pada halaman %d"%(year,month,day,ids))
                break
            soup = BeautifulSoup(r.content,'html.parser')
            titleList = [I.getText() for I in soup.find_all('a','articles--rows--item__title-link')]
            urlList = [I['href'] for I in soup.find_all('a','articles--rows--item__title-link')]
            timeList = [int(I.getText().split(" ")[-1].replace(':','')) for I in soup.find_all('time')]
            if(len(titleList)==0):
                break
            if len(titleList) == len(timeList):
	            for J in range(len(titleList)):
	                    result.append({'judul':titleList[J],
	                   'url':urlList[J],
	                   'tanggal':date,
	                   'sumber':"Liputan6",
	                   'jam':timeList[J]})
            ids += 1
        return result