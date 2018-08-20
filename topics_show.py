import sys
import os
from spyre import server

from spyre.server import Site, App


from topics_analyze import TopicApp
from topics_keys import TopicNameApp



class Index(App):
    def getHTML(self, params):
        return "Title Page Here"


#site = Site(Index)
if __name__ == '__main__':
        portnum = int(sys.argv[1])
        path = os.path.dirname(os.path.abspath(__file__))
        
        with open(path + os.sep+"args.txt","w+") as f:
                for x in sys.argv[2:]:
                        f.write(x+"\n")
        site = Site(TopicNameApp)
        site.addApp(TopicApp, '/analyze data')


        site.launch(port=portnum)






