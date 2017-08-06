from Scraper import Kompas, MetroTv, Liputan6, Detik
import requests

class FileParser:
    parserList = ["Detik"]
    '''
    FileParser pada dasarnya seperti interface di Java.
    Fungsinya untuk mempermudah dan meuniversalkan akses ke scraper untuk koran spesifik

    NO:
    >>>import Kompas
    >>>import Liputan6
    >>>(import dst)
    >>>kompas = Kompas()
    >>>liputan6 = Liputan6()
    >>>(init dst)
    >>>kompas.extractData()

    YES:
    >>>import FileParser
    >>>kompas = FileParser("Kompas")
    >>>liputan6 = FileParser("Liputan6")
    >>>(init dst)
    >>>kompas.extractData()
    '''
    def __init__(self,sumber):
        '''
        Menginiliasi FileParser untuk koran yang mana
        INPUT:

        self --
        sumber -- string
        sumber adalah sebuah string yang menunjukan scraper-spesifik
        mana yang ingin diinisiliasi
        
        Mode yang tersedia:
        Kompas -- kompas.com
        MetroTv - metrotvnews.com

        EXCEPTION:
        ValueError() jika sumber tidak ditemukan

        Jika ingin membuat kelas scraper-spesifik untuk koran lain maka fungsi
        __init__ di bawah harus diubah agar bisa menyetel 
        '''

        # self.__data menampung fungsi extractData() dan extractPublikasi()
        # dari setiap spesifik scraper
        # self.__data['data'] = fungsi mengekstrakData jika diberikan htmlFile dari situs tersebut
        # self.__data['publikasi'] = fungsi mengekstrak publikasi jika diberikan tanggal dari situs tersebut
        if sumber == "Kompas":
            self.__data = {'data':Kompas.Kompas.extractData,'publikasi':Kompas.Kompas.extractPublikasi}
        elif sumber == "MetroTv":
            self.__data = {'data':MetroTv.MetroTv.extractData,'publikasi':MetroTv.MetroTv.extractPublikasi}
        elif sumber == "Liputan6":
            self.__data = {'data':Liputan6.Liputan6.extractData,'publikasi':Liputan6.Liputan6.extractPublikasi}
        elif sumber == "Detik":
            self.__data = {'data':Detik.Detik.extractData,'publikasi':Detik.Detik.extractPublikasi}
        else:
            raise ValueError("%s tidak ditemukan dalam FileParser" % (sumber))
    def extractData(self,htmlFile):
        """
        Mengekstrak isi berita dari situs yang sudah disetting
        INPUT:

        htmlFile -- request.models.Response
        Sebuah request.models.Response yang diekstrak dari laman
        berita yang sudah disetel

        OUTPUT:

        me-return sebuah dictionary berisi berita yang htmlFile 
        yang sudah diekstrak

        Jika ingin membuat kelas scrapper-spesifik untuk koran lain,
        fungsi ekstractData tersebut harus compatible dengan fungsi iini 
        """
        return self.__data['data'](htmlFile)
    def extractPublikasi(self,date):
        """
        Meng-ekstrak semua artikel bisnis keuangan yang
        di hasilkan pada situs yang sudah disetting

        INPUT:

        date -- integer
        Sebuah angka dengan format
        date = year * 10000 + month * 100 + day

        OUTPUT:

        Sebuah list yang isinya dictionary. Setiap dictionary isinya

        sumber : id sumbernya (seperti sumber di __init__)
        tanggal : integer dari variable date
        judul : judul dari artikel
        url : pranala artiker
        jam : jam publikasi dalam bentuk integer dalam format

        jam = 100*h + m

        Jika ingin membuat kelas scrapper-spesifik untuk koran lain,
        fungsi untuk mengesktrak publikasi harus compatible dengan fungsi ini
        """
        return self.__data['publikasi'](date)
    def kompilasiArtikel(self,date):
        """
        Mengekstrak setiap isi berita dari satu tanggal.
        Pada dasarnya. Ini extractPublikasi + extractData

        INPUT:

        date -- integer
        Sebuah angka dengan format
        date = year * 10000 + month * 100 + day

        OUTPUT:

        me-return sebuah list yang berisi list-list  dimana
        list[0] dictionary meta-data berita 
        list[1] string berita html yang sudah diekstrak
        """
        articleList = self.__data['publikasi'](date)
        ids = articleList[0][0]
        result = []
        for I in articleList:
            for J in range(3):
                r = requests.get(I['url'])
                if r.status_code == 200:
                    result.append([I,self.__data['data'](r)])
                    break
            else:
                warn("FileParser gagal mengekstrak %s",(I['url']))
        return result