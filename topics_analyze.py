# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 09:56:34 2016

@author: ani
"""


from spyre import server
server.include_df_index = True

import sys
from os import listdir
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import cherrypy
server.cherrypy.response.timeout = 300000
server.cherrypy.config.update({'response.timeout': 300000})

cherrypy.response.timeout = 3000000
cherrypy.response.timed_out= False

from bidi import algorithm as bidialg
import time
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import itertools

class TopicApp(server.App):
    
    #helper functions        
    def findsep(self,path):
        #find the seperator of path(\ or /)'
        if path == path.split('\\')[0]:
            return '/'
        else:
            return '\\'
        
    def getlabelsval(init,paths,reCol,sep):
        labels = np.core.defchararray.rsplit(paths,sep)
        labels = [np.array(lab)[reCol] for lab in labels]
        labels = [x.tolist() for x in labels]
        return sorted(list(set(labels)))
    def get_topics_name(self):
        path = self.path
        topicnamefile =  'topic_name.txt'
        #get name of topics from topicnamefile is exist or define it to be range of topic num
        if not topicnamefile in listdir(path):
            with open(path + os.sep + topicnamefile,"w") as f:
                topics = map(str,range(len(self.words)))
                for i in topics:
                    f.write(str(i)+"\n")
        with open(path + os.sep + topicnamefile) as f:
            self.topics = f.readlines()
    def __init__(self,path=None,maxfilenum=None):
        sep = os.sep
        if path is None and maxfilenum is None:
            sep = os.sep
            p = os.path.dirname(os.path.abspath(__file__))
            with open(p + sep + "args.txt","r") as f:
                path = f.readline()[:-1]
                maxfilenum = int(f.readline()[:-1])
        self.path = path
        self.maxfilenum = maxfilenum
        #var path include path to directory with files of app

        #vriables to store kmeans results
        self.clusdic = {}

        #inputs - html list to get data from user
        self.inputs = []
        self.params = None
        
        #names of files in directory
        compfile = 'composition.txt'
        keysfile = 'keys.txt'
        sizeoffiles = 'size_of_files.txt'
        labelsfile = 'labels_dic.txt'

        #get data from compfile
        if compfile in listdir(path):
            #labels - path to file. data - TM results of file
            labels = np.genfromtxt(path + sep + compfile, delimiter='\t', usecols=1, dtype=str)
            data = np.genfromtxt(path + sep + compfile, delimiter='\t')[:,2:]
            labels = [labels[i][5:] for i in range(len(labels))]
            self.rdata = []
            
        #get words of topics from keysfile
        if keysfile in listdir(path):
            self.words = np.genfromtxt(path + sep + keysfile , delimiter='\t', usecols=2,dtype =str)
        self.get_topics_name()
        
        labeldic = {}
        reCol = []
        i = 0
        #get the labels that diffrent filse.
        if labelsfile in listdir(path):
            with open(path + sep + labelsfile) as f:
                key,val = f.readline().split('\t')
                if key == 'Title':
                    self.title = val
                else:
                    labeldic[i] = val
                    reCol.append(int(key))
                    i = i+1
                for line in f:
                    key,val = line.split('\t')
                    labeldic[i] = val
                    reCol.append(int(key))
                    i = i+1
        self.labeldic = labeldic
        self.sep = self.findsep(labels[0])
        #get size of files
        if sizeoffiles in listdir(path):
            sizes = np.genfromtxt(path + sep + sizeoffiles, delimiter='\t', usecols=0)
            paths = np.genfromtxt(path + sep + sizeoffiles, delimiter='\t', usecols=1,dtype =str)
            #normalize data by size of files. only files with siz will take into account.
            data = [np.append(data[i],[1]) for i in range(len(data)) if (labels[i] == paths).any()]
            labels = [labels[i] for i in range(len(labels)) if (labels[i] == paths).any()]
            data = [data[i]*int(sizes[paths==labels[i]][0]) for i in range(len(data))]
            #get labels of data
            alabels = np.core.defchararray.rsplit(labels,self.sep)
            alabels = [np.array(lab)[reCol] for lab in alabels]
            alabels = [tuple(x) for x in alabels]
            kalabels = sorted(list(set(alabels)))
            datadic ={}
            labelsdic ={}
            sizedic = {}
            for k in kalabels:
                datadic[k]=[]
                labelsdic[k] = []
            for ind in range(len(labels)):
                datadic[alabels[ind]].append(data[ind])
                labelsdic[alabels[ind]].append(labels[ind])
            for k in datadic.keys():
                sizedic[k] = [d[-1] for d in datadic[k]]
                datadic[k] = [d[:-1] for d in datadic[k]]
            self.datadic =  datadic
            self.sizedic = sizedic
            self.labelsdic = labelsdic
            rdata = [d[:-1] for d in data]
            self.data = rdata
            self.labels = np.array(labels)
            sizes = [d[-1] for d in data]
            self.sizes = np.array(sizes)
        labelvalue = {k:self.getlabelsval(self.labels,reCol[k],self.sep) for k in range(len(reCol))}
    
        #define user input methods        
        filterinput = [{	"type":'checkboxgroup',
				"label": 'fiter data by '+ labeldic[k],  
				"options" : [{"label":v,"value":v,"checked":False} for v in va],
				"key": 'filter_label'+str(k), 
				"action_id" : "refresh",
			} for k,va in labelvalue.iteritems()]
        colurout = [{	"type":'radiobuttons',
				"label": 'differiate data by',  
				"options" : [{"label":v,"value":k,"checked":k==labeldic.keys()[0]} for k,v in labeldic.iteritems()],
				"key": 'get_relevant', 
				"action_id" : "refresh",
			}]
        pcaslider = [{ "type":"slider",
                          "label":"num of pca vectors to show",
                          "key":"pca_arrows",
                          "value":8, "max":len(self.topics),"min":1,
                        "action_id":"refresh"}]
        kmclusnum = [{ "type":"slider",
                          "label":"num of k-means clusters",
                          "key":"kmeansn",
                          "value":8, "max":min(self.maxfilenum,len(self.data)),"min":1,
                        "action_id":"refresh"}]
        self.inputs = filterinput+colurout+pcaslider+kmclusnum
        
    controls = [{"type" : "hidden",
					"label" : "refresh",
					"id" : "refresh" 
				},
                    {	"type" : "button",
					"id" : "submit_plot",
					"label" : "refresh",
				},
				]
    tabs = ["topics","pie","correlations","bar","pca","t-sne","k-means"]
    #define tabs of the web page
    outputs = [{	"type" : "html",
					"id" : "rel_data",
					"control_id" : "refresh",
                         "tab":"pie"
				},
               {	"type" : "html",
					"id" : "html_out",
					"tab" : "topics",
                        "control_id" : "submit_plot",
                         "on_page_load" : True
				},
                   {	"type" : "html",
					"id" : "kmeanswarn",
					"control_id" : "submit_plot",
                         "tab":"k-means",
                         "on_page_load" : False
				},
                    {   "type" : "plot",
					"id" : "plotkmeans",
                                        "control_id" : "submit_plot",
					"tab" : "k-means",
                         "on_page_load" : False
                        },
                    {	"type" : "html",
					"id" : "kmeanstable",
					"control_id" : "submit_plot",
                         "tab":"k-means",
                         "on_page_load" : True
				},
               {	"type" : "html",
					"id" : "barswarn",
					"control_id" : "submit_plot",
                         "tab":"bar",
                         "on_page_load" : False
				},
                        {"type" : "plot",
					"id" : "plotbar",
					"control_id" : "submit_plot",
					"tab" : "bar",
                         "on_page_load" : False},
                     {"type" : "plot",
                     "id" : "plotpie",
                     "control_id" : "submit_plot",
                     "tab" : "pie",
                         "on_page_load" : False},
                     {"type" : "plot",
					"id" : "plotpca",
					"control_id" : "submit_plot",
					"tab" : "pca",
                         "on_page_load" : False},
               {	"type" : "html",
					"id" : "tsnewarn",
					"control_id" : "submit_plot",
                         "tab":"t-sne",
                         "on_page_load" : False
				},
                     {"type" : "plot",
					"id" : "tsneplot",
					"control_id" : "submit_plot",
					"tab" : "t-sne",
                         "on_page_load" : False},
                     {"type" : "plot",
					"id" : "findcorr",
					"control_id" : "submit_plot",
					"tab" : "correlations",
                         "on_page_load" : False}]
    
    #get relevant data - according to user input
    def rel_data(self, params):
        #if the user input dont change return
        if (self.params is not None) and [params['filter_label'+str(k)] for k in self.labeldic.keys()]+[params['get_relevant']]==[self.params['filter_label'+str(k)] for k in self.labeldic.keys()]+[self.params['get_relevant']]:
            return ' '
        if [] in [params['filter_label'+str(k)] for k in self.labeldic.keys()]:
            return '  '
        label = np.core.defchararray.rsplit(self.labels,self.sep)
        indx = range(len(label))
        rlabels = []
        rleg = []
        for k in self.labeldic.keys():
            val = params['filter_label'+str(k)]
            rlabels.append(val)
        for k in params['get_relevant']:
               val = params['filter_label'+str(k)]
               rleg.append(val)
        leg = itertools.product(*rleg)
        rlab =  itertools.product(*rlabels)
        rleg = []
        rlabels = []
        for i in leg:
            rleg.append(i)
        for i in rlab:
            rlabels.append(i)
        #save data in dictioanries with rel_labels keys
        rdic = {}
        rldic = {}
        sdic = {}
        col = params['get_relevant']
        col = [int(x) for x in col]
        rll = sorted(list(set([tuple(np.array(lab)[np.array(col)]) for lab in rlabels])))
        for r in rll:
                rdic[r] = []
                rldic[r] = []
                sdic[r] = []
        for lab in rlabels:
            #add data to dictionaries by key
            if lab in self.datadic.keys():
                rdic[tuple(np.array(lab)[np.array(col)])] = rdic[tuple(np.array(lab)[np.array(col)])] + self.datadic[lab]
                rldic[tuple(np.array(lab)[np.array(col)])] = rldic[tuple(np.array(lab)[np.array(col)])] + self.labelsdic[lab]
                sdic[tuple(np.array(lab)[np.array(col)])] = sdic[tuple(np.array(lab)[np.array(col)])] + self.sizedic[lab]
        for r in rll:
                  if rdic[r] ==[]:
                #no file with this label
                        rll.remove(r)
                        del(rdic[r])
                        del(rldic[r])
                        del(sdic[r])
        if rll == []:
            #no data
            return '  '
       #save also list of rel-data to compute pca, t-sne enc.
        rdata = []
        rpath = []
        sizes = []
        l = len(rll)
        #hard coded rgb colors
        color = ['rgb(255, 000, 000)','rgb(255, 128, 000)','rgb(255, 255, 000)','rgb(128, 255, 000)','rgb(000, 255, 000)','rgb(000, 255, 128)','rgb(000, 255, 255)','rgb(000, 128, 255)','rgb(000, 000, 255)','rgb(127, 000, 255)','rgb(255, 000, 255)','rgb(255, 000, 127)','rgb(128, 128, 128)','rgb(51, 000, 000)']
        #add rbg color on end of path to color it when display
        if len(rll)<=len(color):
            for j in range(len(rll)):
                    i = rll[j]
                    rdata = rdata + rdic[i]
                    rpath = rpath + [x + color[j] for x in rldic[i]]
                    sizes = sizes + sdic[i]
        else:
            cs = cm.Set1(np.arange(l)/(l+0.001))
            for j in range(len(rll)):
                    i = rll[j]
                    rdata = rdata + rdic[i]
                    rpath = rpath + [x + 'rgb(' + str(int(np.multiply(cs[j][0],255)))[:3].zfill(3) + ', '+ str(int(np.multiply(cs[j][1],255)))[:3].zfill(3)+', ' + str(int(np.multiply(cs[j][2],255)))[:3].zfill(3) + ')'for x in rldic[i]]
                    sizes = sizes + sdic[i]
        self.rll = rll
        ind = 0
        self.rdic = {}
        for k in rll:
            self.rdic[k] = range(ind,ind + len(rdic[k]))
            ind =ind + len(rdic[k])
        self.sdic = sdic    
        self.rpath = rpath
        self.rsizes = sizes
        self.rdata=np.array(rdata)
        self.params = params
        return ""

    #diplay table of words of topics.
    def html_out(self,params):
        self.get_topics_name()
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
                                <td height="30" align="center">    """+topics[i]+"""   </td>
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

    #display warning on max number of clusters
    def kmeanswarn(self, params):
        a = self.rel_data(params)
        if a =='  ':
            return ""
        if len(self.rdata)>self.maxfilenum:
            return """<!DOCTYPE html><html><p>too match data to server. pleas choose less data to analyze.  (choose""" + str(len(self.rdata)) + """file, max num of files is """ + str(self.maxfilenum)+""") </p></html>"""
        return """<!DOCTYPE html><html><p>NOTE: num of k-means clusters must be small then """ + str(len(self.rdata)+1) + """</p></html>"""

    def barswarn(self, params):
        a = self.rel_data(params)
        if a =='  ':
            return ""
        if len(self.rdata)>self.maxfilenum:
            return """<!DOCTYPE html><html><p>too match data to server. pleas choose less data to analyze.  (choose""" + str(len(self.rdata)) + """file, max num of files is """ + str(self.maxfilenum)+""") </p></html>"""
        return ""
    def tsnewarn(self,params):
        return self.barswarn(params)
    #plot html table with paths of files in each k-means cluster
    def kmeanstable(self,params):
        a = self.rel_data(params)
        if a =='  ':
            return ""
        sep = self.sep
        #check if clusters of this inputs computed
        if str([params['filter_label'+str(k)] for k in self.labeldic.keys()]+[params['kmeansn']]+[params['get_relevant']]) in self.clusdic.keys():      
                return self.clusdic[str([params['filter_label'+str(k)] for k in self.labeldic.keys()]+[params['kmeansn']]+[params['get_relevant']])][1]
        #cumpute k-means
        kmeansclus = int(params['kmeansn'])
        clusters = KMeans(n_clusters=kmeansclus).fit(self.rdata)
        clusters = clusters.labels_

        table = """ <thead>
            <tr>
                <th>path to file</th>
                <th>cluster num</th>
            </tr>
        </thead>"""
        #add pathes to table
        rl = np.array(self.rpath)
        for i in range(kmeansclus):
            pathtable = """<table>"""
            clus = rl[i==clusters]
            for j in range(len(clus)):
                #color paths by labels
                pathtable = pathtable + """<tr>
                                                <td height="30">
                                                    <font color = """ + clus[j][-18:] +""">"""+clus[j][:-18] + """</font>
                                                   </td>
                                        </tr>"""
            pathtable =  pathtable+ """</table>"""
            table = table + """<tr><td>"""+ pathtable + """</td> <td> """+str(i) +"""</td>  </tr>"""
            self.clusdic[str([params['filter_label'+str(k)] for k in self.labeldic.keys()]+[params['kmeansn']]+[params['get_relevant']])] = [clusters, """<!DOCTYPE html>
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
 

        <table id="myTable">"""+table + """
        </table>
    </body>
</html>"""]
        return self.clusdic[str([params['filter_label'+str(k)] for k in self.labeldic.keys()]+[params['kmeansn']]+[params['get_relevant']])][1]

    #plot data on clusters
    def plotkmeans(self,params):
        a = self.rel_data(params)
        if a =='  ':
            return plt.figure()
        if len(self.rdata)>self.maxfilenum:
            return plt.figure()
        #get clusters
        while str([params['filter_label'+str(k)] for k in self.labeldic.keys()]+[params['kmeansn']]+[params['get_relevant']]) not in self.clusdic.keys():
            time.sleep(2)
        clusters = self.clusdic[str([params['filter_label'+str(k)] for k in self.labeldic.keys()]+[params['kmeansn']]+[params['get_relevant']])][0]
        rdata = np.array(self.rdata)
        #plot ditribution of topics in each clusters by bars graph
        scor = []
        lab = []
        size = []
        #num of clusters
        kn = int(params['kmeansn'])
        l = len(self.topics)
        d = pd.DataFrame(index =range(kn),columns = range(l))
        fig, axes = plt.subplots(1, 2, figsize=(17, 7));
        cs = cm.Set1(np.arange(l)/(l+0.001))
        #get size of each clusters (in bytes)
        siz = np.array(self.rsizes)
        for i in range(kn):
                size = size + [np.sum(siz[i==clusters])]
        m = np.max(size)
        #compute ditribution of topics in each cluster
        for i in range(kn):
            clus = rdata[i==clusters]
            scor = np.sum(clus,0)
            d = pd.DataFrame(index =range(kn),columns = range(l))
            d.fillna(0)
            d =d.T
            d[i] = scor/np.sum(scor)
            d = d.T
            d.columns = [bidialg.get_display(x) for x in self.topics]
            fig = d.plot(kind='bar',legend = False,stacked=True,color = cs,ax = axes[0],width = 0.8*size[i]/m)
        handles, labelss = axes[0].get_legend_handles_labels()
        axes[0].set_title('bars of topics vs cluster name')
        axes[0].set_xlabel('num of cluster')
        axes[0].legend(handles = handles[0:l], loc='center left', bbox_to_anchor=(1.0, 0.5))
        if not len(params['get_relevant'])>0:
            #cant compute distribution of labels
                axes[1].set_visible(False)
                return axes[0]
        #plot ditribution of labels in each clusters by bars graph
        l = len(self.rll)
        cc = cm.Set1(np.arange(l)/(l+0.001))
        for i in range(kn):
                c = []
                for j in range(l):
                        #get size of label j in cluster i
                        x = np.zeros(len(self.rdata))
                        x.fill(-1)
                        x[self.rdic[self.rll[j]]] = clusters[self.rdic[self.rll[j]]]                        
                        c = c+[float(np.sum(siz[x==i]))]
                d = pd.DataFrame(index =range(kn),columns = range(l))
                d.fillna(0)
                d =d.T
                if np.sum(c) != 0:
                        d[i] = c/np.sum(c)
                d = d.T
                d.columns = [str([str(s) for s in x])[1:-1].replace("'", "") for x in self.rll]
                fig = d.plot(kind='bar',legend = False,stacked=True,color = cc,ax = axes[1],width = 0.8*size[i]/m)
        handles, labelss = axes[1].get_legend_handles_labels()
        axes[1].set_title('bars of committees protocols vs cluster name')
        axes[1].set_xlabel('num of cluster')
        axes[1].legend(handles = handles[0:l], loc='center left', bbox_to_anchor=(1.0, 0.5))
        axes[0].grid(False)
        axes[1].grid(False)
        return fig

    #plot distribution of topics in each labels. if user dont declare labels plot hist of topics
    def plotbar(self,params):
        sep = self.sep
        a = self.rel_data(params)
        if a =='  ':
            return plt.figure()
        if len(self.rdata)>self.maxfilenum:
            return plt.figure()
        l = len(self.topics)
        cs = cm.Set1(np.arange(l)/(l+0.001))
        col = params['get_relevant']
        #plot distribution of topics in each labels.
        label = self.rll 
        bardata = []
        labsize = []
        #get data and size of each label
        for i in label:
            bardata.append(self.rdata[self.rdic[i]])                
            labsize.append(np.sum(self.sdic[i]))
            m = np.max(labsize)
        bardata = [np.sum(i,0) for i in bardata]
        bardata = [bardata[i]/labsize[i] for i in range(len(bardata))]
        bardata = [x/np.sum(x) for x in bardata]
        c = pd.DataFrame(bardata).T
        mp = {i:str([str(s) for s in self.rll[i]])[1:-1].replace("'", "") for i in range(len(label))}
        d = pd.DataFrame(index = range(l),columns = range(len(label)))
        d= d.fillna(0)
        ax = d.plot(legend = False,figsize = (17,8.5))
        for i in range(len(label)):
            #create bar
            d = pd.DataFrame(index = range(l),columns = range(len(label)))
            d = d.fillna(0)
            d[i] = c[i]
            d  = d.T
            d = d.rename(index = mp)
            d.columns = [bidialg.get_display(x) for x in self.topics]
            #width of bar indicate its size
            fig = d.plot(kind='bar',legend = False,stacked=True,figsize = (17,8.5),color = cs,width= 0.85*labsize[i]/m,ax = ax)
        handles, labelss = ax.get_legend_handles_labels()
        fig.legend(handles = handles[len(label):l+len(label)], loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.grid(False)
        return ax

    #plot sistibution of topics in relevant data in pie graph
    def plotpie(self,params):
        a = self.rel_data(params)
        if a =='  ':
            return plt.figure()
        scors = np.sum(self.rdata,0)
        scors = scors/np.sum(self.rsizes)
        l = len(scors)        
        cs = cm.Set1(np.arange(l)/(l+0.001))
        fig = plt.figure()       
        ax = fig.add_subplot(111,aspect='equal')
        ax.pie(scors, labels = [bidialg.get_display(x) for x in self.topics], colors=cs)
        return fig

    #plot corralation between topics in rel-data
    def findcorr(self,params):
        a = self.rel_data(params)
        if a =='  ':
            return plt.figure()
        d = pd.DataFrame(data = self.rdata,columns = [bidialg.get_display(x) for x in self.topics])
        cor = d.corr()
        mask = np.zeros_like(cor, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # St up the matplotlib figure
        f, ax = plt.subplots(figsize=(11, 9))
        # Gnerate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        b = sns.heatmap(cor, mask=mask, cmap=cmap, vmax=.3,
                square=True, xticklabels=True, yticklabels=True,linewidths=.5,cbar_kws={"shrink": .5},ax=ax)
        for i in b.get_xticklabels():
            i.set_rotation(90)
        for i in b.get_yticklabels():
            i.set_rotation(360)
        return ax

    #calculate 2d-pca of data and plot it.
    def plotpca(self,params):
        sep = self.sep
        a = self.rel_data(params)
        if a =='  ':
            return plt.figure()
        pca = PCA(n_components = 2)
        #calaulate pca
        p = pca.fit(self.rdata)
        pc = p.transform(self.rdata)
        coef = p.components_
        coefsize = [(coef[0,i])*(coef[0,i])+ (coef[1,i])*(coef[1,i]) for i in range(len(coef[0,:]))]
        
        fig = plt.figure(figsize = (14,7))
        ax = fig.add_subplot(121)
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        keys = np.array(self.topics)
        #plot the biggest coef vectors of pca results
        if params['pca_arrows']<len(self.topics):
            n = int(len(self.topics) - params['pca_arrows']- 1)
            minn = np.partition(coefsize,n)[n]
            rcoe = np.array([coe>minn for coe in coefsize])
            coef = coef[:,rcoe]
            keys = keys[rcoe]
        for i in range(len(coef[0,:])):
            ax.arrow(0,0,coef[0,i],coef[1,i],alpha =0.5,clip_on=False)
            ax.text(coef[0,i]*1.15,coef[1,i]*1.15,bidialg.get_display(keys[i]), ha='center', va='center')

        #plot pca data
        ax2 = fig.add_subplot(122)
        labels = self.rll
        l = len(labels)
        cs = cm.Set1(np.arange(l)/(l+0.001))

          #plot labels in diffrent colors
        for i in range(l):
            lab = labels[i]
            lableg = [str(llab) for llab in labels[i]]
            ax2.plot(pc[self.rdic[labels[i]],0],pc[self.rdic[labels[i]],1],'o',ms = 4.0,color = cs[i],label = str(lableg)[1:-1].replace("'", ""))         
        ax2.legend(loc='center left', bbox_to_anchor=(0.95, 0.5),numpoints=1)
        ax2.set_xlabel('1st Principa Component')
        ax2.set_ylabel('2nd Principa Component')
         
        ax2.title.set_text('pca of data')
        return fig

    #plot t-sne of data
    def tsneplot(self,params):
        a = self.rel_data(params)
        if a =='  ':
            return plt.figure()
        if len(self.rdata)>self.maxfilenum:
            return plt.figure()
        sep = self.sep
        #calculate t-sne
        ts = TSNE(n_components = 2)
        tscore = ts.fit_transform(self.rdata)
        #plot results
        f = plt.figure()
        r = 0.065,0.125,0.75,0.75
        ax = f.add_axes(r)
        col = params['get_relevant']
        #plot labels in diffrent colors
        col = [int(x) for x in col]
        rleg = self.rll
        l = len(self.rll)
        cs = cm.Set1(np.arange(l)/(l+0.001))
        for i  in range(l):
            lab = rleg[i]
            lableg = [str(llab) for llab in lab]
            ax.plot(tscore[self.rdic[lab],0],tscore[self.rdic[lab],1],'o',ms = 3.0,color = cs[i],label = str(lableg)[1:-1].replace("'", ""))
        handles, labelss = ax.get_legend_handles_labels()
        ax.legend(handles = handles,loc='center left', bbox_to_anchor=(1, 0.5),numpoints=1)
        return f

    
    
if __name__ == '__main__':
    #get path to directory with file to app
    portnum = sys.argv[1]
    argpath  = int(sys.argv[2])
    maxfilenum = int(sys.argv[3])
    #init app
    app = TopicApp(argpath,maxfilenum)
    #run
    app.launch(port=portnum)
 
