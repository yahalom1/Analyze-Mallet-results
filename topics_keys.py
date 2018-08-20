
from spyre import server
server.include_df_index = True
import numpy as np

import sys
import os
from os import listdir
import time

class TopicNameApp(server.App):
    def get_data(self):
        sep = os.sep
        keysfile = 'keys.txt'
        topicnamefile =  'topic_name.txt'
        path =self.path
        
        if keysfile in listdir(path):
            self.words = np.genfromtxt(path + sep + keysfile , delimiter='\t', usecols=2,dtype =str)
        #get name of topics from topicnamefile is exist or define it to be range of topic num
        if not topicnamefile in listdir(path):
            with open(path + sep + topicnamefile,"w") as f:
                topics = map(str,range(len(self.words)))
                for i in topics:
                    f.write(str(i)+"\n")
        with open(path + sep + topicnamefile) as f:
            self.topics = f.readlines()
    def __init__(self,path=None):
        if path is None:
            sep = os.sep
            p = os.path.dirname(os.path.abspath(__file__))
            with open(p + sep + "args.txt","r") as f:
                path = f.readline()[:-1]
            
        self.path = path
        self.get_data()
        self.inputs = [{	"type":'text',
				"label": str(i), 
				"value" : "",
				"key": "a"+str(i), 
				"action_id" : "refresh",
			} for i in range(len(self.topics))]
    controls = [
                    {	"type" : "button",
					"id" : "refresh",
					"label" : "save topics name",
				},
				]
    outputs = [{	"type" : "html",
					"id" : "html_out",
					"control_id" : "refresh",
                        "on_page_load" : True
				}
             ]
    title = "Topics show"
   
    def html_out(self,params):
        time.sleep(1)
        self.get_data()
        topicnamefile =  'topic_name.txt'
        topics= []
        for i in range(len(self.topics)):
            topics.append(params["a"+str(i)])
        for i in range(len(topics)):
            if topics[i]=="":
                topics[i] = self.topics[i]
            else:
                topics[i] = topics[i]+"\n"    
        with open(self.path + os.sep + topicnamefile,"w+") as f:
                for i in topics:
                      f.write(str(i))
        self.get_data()
        #get names of topics and there words
        
        topics = self.topics
        words = self.words
        words = [words[i].split(" ")[:-1] for i in range(len(words))]
        for i in range(len(words)):
            for j in range(len(words[0])):
                words[i][j] = words[i][j].replace("\""," ")
        #num to wors to show for each topic.
        nwordsshow = min(len(words[0]),20)
        #creat the table
        table = """ <thead>
            <tr>
                <th>Topic words</th>
                <th>Topic nane</th>
            </tr>
        </thead>"""
        for i in range(len(topics)):
            table = table + """<tr>
                                <td align="right"></td>
                                <td height="30" align="center">    """+topics[i]+"""    </td>
                                <td align="center">"""+str(i) +"""</td>
                            </tr>"""
        alltable = ""
        for i in range(len(words)):
            alltable = alltable + "." + ','.join(words[i])
        alltable = "\"" + alltable[1:] + "\""
        return """<!DOCTYPE html>
                <html>
                <head>
                <meta charset="utf-8">
                    <style>
                        table, td {
                        border: 1px solid black;
                        }
                    </style>
                    </head>
                        <body>
                            <p>change name of topics in text boxes in left and click save to save your name</p>
                            <p>if you don't see ths save buttom try zoom out.</p>
                            

                            
                            <table id="myTable" align="right"><caption align=bottom>Choose num of words to show per topic (between 1 and """ + str(len(words[0])) +"""):<input type="number" id = "myNumber" name="quantity" value = "20" min="1" max=" """ + str(len(words[0])) +""" " style="width: 80px">
                                <button onclick="yFunction()">Go</button></caption>"""+table + """
                            </table>



                            <script>
                                window.onload = yFunction();
                                function yFunction() {
                                    var table = """ + alltable + """;
                                    table = table.split(".")
                                    var num = document.getElementById("myNumber").value;
                                    var oTable = document.getElementById('myTable');
                                    var rowLength = oTable.rows.length;  
                                    for (i = 0; i < rowLength-1; i++){
                                        var s = table[i].split(",",num);
                                        oTable.rows[i+1].cells[0].innerHTML = s.join().replace(/ /g,"\\"").replace(/,/g,", ");
                                    }
                                }
                            </script>

                        </body>
                </html>"""


      
if __name__ == '__main__':
    #get path to directory with file to app
    portnum = sys.argv[1]
    argpath = int(sys.argv[2])
    #init app
    app = TopicNameApp(argpath)
    #run
    app.launch(port=portnum)
 

    
