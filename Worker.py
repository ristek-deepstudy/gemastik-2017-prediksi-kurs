import queue
import threading
import requests

class Worker:
    def __init__(self):
        self.results = {}
        self.q = queue.Queue()
        self.count = 0
        for u in range(50):
            t = threading.Thread(target=self.getUrl)
            t.daemon = True
            t.start()
    def getUrl(self):
        while True:
            url = self.q.get()
            for I in range(3):
                try:
                    r = requests.get(url)
                    if r.status_code == 200:
                        self.results[url] = r
                except:
                    continue
            self.q.task_done()
            self.count -= 1
    def reset(self):
        self.results = {}
    def addOrder(self,order):
            self.q.put(order)
            self.count += 1
    def getData(self):
        while True:
            if self.count == 0:
                return self.results