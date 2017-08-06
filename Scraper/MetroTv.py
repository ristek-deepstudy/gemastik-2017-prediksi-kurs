import requests
from bs4 import BeautifulSoup
from warnings import warn
class MetroTv:
    def extractData(htmlFile):
        """
        Mengekstrak isi berita kompas.com
        INPUT:
        
        htmlFile -- request.models.Response
        Sebuah request.models.Response yang diekstrak dari laman
        berita kompas.com
        
        OUTPUT:
        
        me-return sebuah string yang berisi berita yang htmlFile kompas.com
        yang sudah diekstrak
        """
        try:
            soup = BeautifulSoup(htmlFile.content, 'html.parser')
            text = soup.find('div',"tru").getText().replace('\r\n','\n')
            text = text.replace("\n \n","\n")
            text = text.split("\n\n\n\n")
            text[0] = text[0].split("Baca juga")[0]
            text = " ".join(text)
            return text.replace('\n\nMetrotvnews.com, ','')
        except:
            return None
    def extractPublikasi(date):
        """
        Meng-ekstrak semua artikel bisnis keuangan yang
        di hasilkan pada tanggal itu di Kompas.com
        
        INPUT:
        
        date -- integer
        Sebuah angka dengan format
        date = year * 10000 + month * 100 + day
        
        OUTPUT:
        
        Sebuah list yang isinya dictionary. Setiap dictionary isinya
        
        sumber : "MetroTv"
        tanggal : integer dari variable date
        judul : judul dari artikel
        url : pranala artiker
        jam : jam publikasi dalam bentuk integer dalam format
        
        jam = 100*h + m
        """
        year = date // 10000
        month = date // 100 % 100
        day = date % 100    
        url = "http://ekonomi.metrotvnews.com/index/%d/%d/%d/%d"
        ids = 0
        result = []
        while True:
            urlList = []
            titleList = []
            # Sebenarnya metrotvnews.com menghasilkan 30 entiti per halaman indeks
            # tetapi suka bug dan menghasilkan 31 entite per halaman indeks dengan
            # entiti yang ekstra muncul lagi di halaman indeks selanjutnya
            for J in range(3):
                r = requests.get(url%(year,month,day,ids*30))
                if r.status_code == 200:
                    break
            else:
                warn("MetroTV gagal mengesktrak indeks pada %d/%d/%d pada halaman %d",(year,month,day,ids+1))
                break
            soup = BeautifulSoup(r.content, 'html.parser')
            dateList = soup.find_all('div','reg')
            if len(dateList) == 0:
                break
            dateList = [int(I.getText().split(" ")[-1].replace(":","")) for I in dateList]
            data = soup.find_all('h2')
            dataNew = []
            for I in data:
                if len(I.find_all('a'))>0:
                    urlList.append(I.find('a')['href'])
                    titleList.append(I.find('a').getText())
            for I in range(len(urlList)):
                result.append({'sumber':'MetroTv','tanggal':date,'judul':titleList[I],
                            'url':urlList[I],'jam':dateList[I]})
            ids += 1

        return result